/* spectrum.c: how low-rank is a Q8_0 weight matrix? The 3090 Ti forms its Gram matrix in OpenCL;
 * LAPACK takes the eigenvalues on the host. Offline analysis of the weights: nothing here runs
 * during inference, and the model's math stays in src/apl/qwen35.apl.
 *
 *   spectrum MODEL OFFSET COLS ROWS NAME     one matrix, from gguf_info's numbers
 *   spectrum --control COLS ROWS             a random Gaussian matrix of that shape
 *
 * Prints, for rank r, the share of W's energy (the sum of squared singular values) that the best
 * rank-r approximation keeps, and its relative error ||W - W_r||_F / ||W||_F (Eckart-Young). A real
 * matrix that looks like the control has no structure a factorization could use.
 *
 * Each GPU job is one row of the Gram matrix, milliseconds, so the desktop, which shares this GPU,
 * keeps running. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK 32
#define BLOCK_BYTES 34 /* f16 scale, then 32 int8 */

extern void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a, const int *lda, float *w, float *work, const int *lwork, int *iwork,
                    const int *liwork, int *info, size_t jobz_len, size_t uplo_len);

static const char *KERNEL =
    "kernel void gram_row(global const float *w, global float *g, const long k, const long len, const long i)\n"
    "{\n"
    "    const long j = get_global_id(0);\n"
    "    global const float *a = w + i * len, *b = w + j * len;\n"
    "    float s = 0;\n"
    "    for (long r = 0; r < len; r++)\n"
    "        s += a[r] * b[r];\n"
    "    g[i * k + j] = s;\n"
    "}\n";

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

__attribute__((noreturn)) static void die(const char *what, int code)
{
    fprintf(stderr, "spectrum: %s: %d\n", what, code);
    exit(1);
}

static float f16_to_f32(uint16_t h)
{
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(m + 1024), e - 25);
    return s ? -v : v;
}

/* W as k vectors of length len: the smaller side's dimension is k, so the Gram matrix is k x k */
static float *load_q8(const char *path, long offset, long cols, long rows, long k, long len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }
    const long row_bytes = (cols / BLOCK) * BLOCK_BYTES;
    unsigned char *row = malloc((size_t)row_bytes);
    float *w = malloc(sizeof(float) * (size_t)k * (size_t)len);
    if (!row || !w || fseek(f, offset, SEEK_SET))
        die("load", 0);
    for (long r = 0; r < rows; r++) {
        if (fread(row, (size_t)row_bytes, 1, f) != 1)
            die("short read", (int)r);
        for (long b = 0; b < cols / BLOCK; b++) {
            const unsigned char *p = row + b * BLOCK_BYTES;
            const float d = f16_to_f32((uint16_t)(p[0] | p[1] << 8));
            for (long c = 0; c < BLOCK; c++) {
                const float v = d * (float)(int8_t)p[2 + c];
                const long col = b * BLOCK + c;
                w[cols <= rows ? col * len + r : r * len + col] = v;
            }
        }
    }
    free(row);
    fclose(f);
    return w;
}

/* Gaussian by Box-Muller from a fixed generator: the control is the same every run */
static float *load_control(long k, long len)
{
    float *w = malloc(sizeof(float) * (size_t)k * (size_t)len);
    if (!w)
        die("control", 0);
    uint64_t s = 0x9E3779B97F4A7C15u;
    for (long i = 0; i < k * len; i++) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const double u = ((double)(s >> 40) + 0.5) / 16777216.0, v = (double)((s >> 8) & 0xFFFFFFu) / 16777216.0;
        w[i] = (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
    }
    return w;
}

static float *gram_on_gpu(const float *w, long k, long len)
{
    cl_int e;
    cl_platform_id plat;
    cl_device_id dev;
    if ((e = clGetPlatformIDs(1, &plat, NULL)) || (e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL)))
        die("no GPU", e);
    char name[128];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    if (e)
        die("context", e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    if (e)
        die("queue", e);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &KERNEL, NULL, &e);
    if (e || (e = clBuildProgram(prog, 1, &dev, "", NULL, NULL)))
        die("build", e);
    cl_kernel kern = clCreateKernel(prog, "gram_row", &e);
    if (e)
        die("kernel", e);
    const size_t w_bytes = sizeof(float) * (size_t)k * (size_t)len, g_bytes = sizeof(float) * (size_t)k * (size_t)k;
    cl_mem bw = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, w_bytes, (void *)w, &e);
    if (e)
        die("weights buffer", e);
    cl_mem bg = clCreateBuffer(ctx, CL_MEM_READ_WRITE, g_bytes, NULL, &e);
    if (e)
        die("gram buffer", e);
    const cl_long lk = k, llen = len;
    clSetKernelArg(kern, 0, sizeof bw, &bw);
    clSetKernelArg(kern, 1, sizeof bg, &bg);
    clSetKernelArg(kern, 2, sizeof lk, &lk);
    clSetKernelArg(kern, 3, sizeof llen, &llen);
    const double t0 = now_s();
    for (long i = 0; i < k; i++) {
        const cl_long li = i;
        const size_t items = (size_t)i + 1; /* the lower triangle */
        clSetKernelArg(kern, 4, sizeof li, &li);
        if ((e = clEnqueueNDRangeKernel(q, kern, 1, NULL, &items, NULL, 0, NULL, NULL)))
            die("enqueue", e);
        if ((i & 15) == 15)
            clFinish(q); /* never a long queue ahead of the desktop */
    }
    clFinish(q);
    float *g = malloc(g_bytes);
    if (!g || (e = clEnqueueReadBuffer(q, bg, CL_TRUE, 0, g_bytes, g, 0, NULL, NULL)))
        die("read", e);
    printf("  %s: Gram %ld x %ld over vectors of %ld: %.2f s\n", name, k, k, len, now_s() - t0);
    clReleaseMemObject(bw);
    clReleaseMemObject(bg);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return g;
}

/* The eigenvalues, ascending; the stored triangle is the lower one in Fortran's terms */
static float *eigenvalues(float *g, long k)
{
    const int n = (int)k;
    int info, liwork = -1, lwork = -1, iq;
    float wq;
    float *ev = malloc(sizeof(float) * (size_t)k);
    if (!ev)
        die("eigenvalues", 0);
    ssyevd_("N", "U", &n, g, &n, ev, &wq, &lwork, &iq, &liwork, &info, 1, 1);
    lwork = (int)wq;
    liwork = iq;
    float *work = malloc(sizeof(float) * (size_t)lwork);
    int *iwork = malloc(sizeof(int) * (size_t)liwork);
    if (!work || !iwork)
        die("workspace", 0);
    const double t0 = now_s();
    ssyevd_("N", "U", &n, g, &n, ev, work, &lwork, iwork, &liwork, &info, 1, 1);
    if (info)
        die("ssyevd", info);
    printf("  LAPACK eigenvalues on the host: %.2f s\n", now_s() - t0);
    free(work);
    free(iwork);
    return ev;
}

static void report(const char *name, const float *ev, long k, long rows, long cols)
{
    double total = 0;
    for (long i = 0; i < k; i++)
        total += ev[i] > 0 ? (double)ev[i] : 0;
    printf("%s: %ld x %ld, top singular values", name, rows, cols);
    for (long i = 0; i < 4; i++)
        printf(" %.3f", sqrt((double)ev[k - 1 - i]));
    printf("; median %.3f; smallest %.3f\n", sqrt((double)ev[k / 2]), sqrt(fabs((double)ev[0])));
    static const double frac[] = {0.0625, 0.125, 0.25, 0.375, 0.5, 0.75};
    for (size_t f = 0; f < sizeof frac / sizeof frac[0]; f++) {
        const long r = (long)(frac[f] * (double)k);
        double kept = 0;
        for (long i = 0; i < r; i++)
            kept += ev[k - 1 - i] > 0 ? (double)ev[k - 1 - i] : 0;
        const double bytes = (double)(rows + cols) * (double)r / ((double)rows * (double)cols);
        printf("  rank %5ld (%5.1f %% of the side): keeps %6.2f %% of the energy, relative error %.4f; factors are %.3f of W's weights\n", r, 100 * frac[f],
               100 * kept / total, sqrt(fmax(0.0, 1 - kept / total)), bytes);
    }
}

int main(int argc, char **argv)
{
    long cols, rows;
    float *w;
    const char *name;
    if (argc == 4 && strcmp(argv[1], "--control") == 0) {
        cols = atol(argv[2]);
        rows = atol(argv[3]);
        name = "control (random Gaussian)";
        const long k = cols <= rows ? cols : rows, len = cols <= rows ? rows : cols;
        w = load_control(k, len);
    } else if (argc == 6) {
        cols = atol(argv[3]);
        rows = atol(argv[4]);
        name = argv[5];
        const long k = cols <= rows ? cols : rows, len = cols <= rows ? rows : cols;
        w = load_q8(argv[1], atol(argv[2]), cols, rows, k, len);
    } else {
        fprintf(stderr, "usage: spectrum MODEL OFFSET COLS ROWS NAME | spectrum --control COLS ROWS\n");
        return 2;
    }
    const long k = cols <= rows ? cols : rows, len = cols <= rows ? rows : cols;
    float *g = gram_on_gpu(w, k, len);
    free(w);
    float *ev = eigenvalues(g, k);
    report(name, ev, k, rows, cols);
    free(g);
    free(ev);
    return 0;
}
