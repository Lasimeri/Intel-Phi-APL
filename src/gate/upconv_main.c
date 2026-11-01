/* upconv_main.c: gate 2.1, the program. Built twice like fma_main.c: by
 * knc-cc against the card kernels and by the host cc against their host
 * model. Never includes knc.h (x87 ABI side).
 *
 *   upconv OUT.bin   run the seven kernels of upconv_kernel.c over the
 *                    inputs of upconv_common.h, write OUT.bin, print one
 *                    hash line per kernel
 *   upconv --probe   the alignment contract: each up-converting kernel once
 *                    at an offset that keeps its operand alignment and once
 *                    at one that breaks it, each in a forked child; one line
 *                    per probe with the child's exit status or signal */
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gate_kernels.h"
#include "upconv_common.h"

static struct g21_in in;
static struct g21_out out;

static int write_results(const char *path)
{
    unsigned char h[64];
    struct g21_array a[7];
    int k = g21_arrays(&out, a);
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return 1;
    }
    g21_header(h);
    int ok = fwrite(h, 1, 64, f) == 64;
    for (int j = 0; j < k; j++)
        ok &= fwrite(a[j].p, sizeof(float), (size_t)a[j].n, f) == (size_t)a[j].n;
    ok &= fclose(f) == 0;
    if (!ok)
        fprintf(stderr, "%s: short write\n", path);
    return !ok;
}

/* Probe buffers: 64-byte aligned, every lane a valid input. */
static unsigned char pb_w[1024] A64;
static float pb_x[16] A64, pb_c[16] A64, pb_o[16] A64;

enum { P_S8, P_U8, P_F16, P_B16 };
static const struct probe {
    const char *name;
    int kernel, offset; /* offset in bytes from a 64-byte boundary */
} probes[] = {
    {"s8    {sint8} load, 16 B operand", P_S8, 16},  {"s8    {sint8} load, 16 B operand", P_S8, 1},
    {"u8fma {uint8} fma,  16 B operand", P_U8, 16},  {"u8fma {uint8} fma,  16 B operand", P_U8, 1},
    {"f16   {float16} fma, 32 B operand", P_F16, 32}, {"f16   {float16} fma, 32 B operand", P_F16, 2},
    {"b16   {1to16} fma,   4 B operand", P_B16, 4},   {"b16   {1to16} fma,   4 B operand", P_B16, 1},
};

static void probe_run(const struct probe *p)
{
    const void *w = pb_w + p->offset;
    switch (p->kernel) {
    case P_S8: k_s8((const int8_t *)w, pb_x, pb_c, pb_o, 16); break;
    case P_U8: k_u8fma((const uint8_t *)w, pb_x, pb_c, pb_o, 16); break;
    case P_F16: k_f16((const uint16_t *)w, pb_x, pb_c, pb_o, 16); break;
    default: k_b16((const float *)w, pb_x, pb_c, pb_o, 16); break;
    }
}

static int probe_all(void)
{
    int bad = 0;
    for (int i = 0; i < 16; i++)
        pb_x[i] = pb_c[i] = 1.0f;
    for (size_t i = 0; i < sizeof probes / sizeof probes[0]; i++) {
        const struct probe *p = &probes[i];
        fflush(stdout);
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            probe_run(p);
            _exit(0);
        }
        int st = 0;
        if (waitpid(pid, &st, 0) != pid) {
            perror("waitpid");
            return 1;
        }
        if (WIFSIGNALED(st))
            printf("probe %s +%-2d signal %d (%s)\n", p->name, p->offset, WTERMSIG(st), strsignal(WTERMSIG(st)));
        else
            printf("probe %s +%-2d exit %d\n", p->name, p->offset, WEXITSTATUS(st));
        bad |= !WIFSIGNALED(st) && WEXITSTATUS(st) != 0;
    }
    return bad;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--probe") == 0)
        return probe_all();
    if (argc != 2 || argv[1][0] == '-') {
        fprintf(stderr, "usage: %s OUT.bin | --probe\n", argv[0]);
        return 2;
    }
    g21_fill(&in);
    k_s8(in.q, in.x8, in.c8, out.s8, G21_N8);
    k_u8fma(in.u, in.x8, in.c8, out.u8fma, G21_N8);
    k_u8mul(in.u, in.x8, in.c8, out.u8mul, G21_N8);
    k_b16(in.s, in.xb, in.cb, out.b16, G21_NB);
    k_bcast(in.s, in.xb, in.cb, out.bcast, G21_NB);
    k_f16(in.h, in.xf, in.cf, out.f16, G21_NF);
    k_hsum(in.xh, out.hsum, G21_NH);
    if (write_results(argv[1]))
        return 1;
    g21_summary(stdout, &out);
    return 0;
}
