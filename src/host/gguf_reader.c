/* gguf_reader.c: see gguf_reader.h. */
#include "gguf_reader.h"

#include <errno.h>
#include <inttypes.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct gtype {
    const char *name;
    uint32_t blck, size;
};

/* Indexed by ggml_type id; a zero block marks an id with no type. */
static const struct gtype gtypes[GGUF_MAX_TENSOR_TYPE] = {
    [0] = {"F32", 1, 4},       [1] = {"F16", 1, 2},       [2] = {"Q4_0", 32, 18},
    [3] = {"Q4_1", 32, 20},    [6] = {"Q5_0", 32, 22},    [7] = {"Q5_1", 32, 24},
    [8] = {"Q8_0", 32, 34},    [9] = {"Q8_1", 32, 36},    [10] = {"Q2_K", 256, 84},
    [11] = {"Q3_K", 256, 110}, [12] = {"Q4_K", 256, 144}, [13] = {"Q5_K", 256, 176},
    [14] = {"Q6_K", 256, 210}, [15] = {"Q8_K", 256, 292}, [16] = {"IQ2_XXS", 256, 66},
    [17] = {"IQ2_XS", 256, 74}, [18] = {"IQ3_XXS", 256, 98}, [19] = {"IQ1_S", 256, 50},
    [20] = {"IQ4_NL", 32, 18}, [21] = {"IQ3_S", 256, 110}, [22] = {"IQ2_S", 256, 82},
    [23] = {"IQ4_XS", 256, 136}, [24] = {"I8", 1, 1},     [25] = {"I16", 1, 2},
    [26] = {"I32", 1, 4},      [27] = {"I64", 1, 8},      [28] = {"F64", 1, 8},
    [29] = {"IQ1_M", 256, 56}, [30] = {"BF16", 1, 2},     [34] = {"TQ1_0", 256, 54},
    [35] = {"TQ2_0", 256, 66}, [39] = {"MXFP4", 32, 17},  [40] = {"NVFP4", 64, 36},
    [41] = {"Q1_0", 128, 18},  [42] = {"Q2_0", 64, 18},
};

const char *gguf_tensor_type_name(uint32_t type)
{
    return type < GGUF_MAX_TENSOR_TYPE && gtypes[type].blck ? gtypes[type].name : NULL;
}

const char *gguf_value_type_name(uint32_t type)
{
    static const char *n[] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "str", "arr", "u64", "i64", "f64"};
    return type < 13 ? n[type] : "?";
}

/* The parse state; any read problem jumps back to gguf_load with a message. */
struct reader {
    FILE *f;
    jmp_buf fail;
    const char *msg;
};

static void bad(struct reader *r, const char *msg)
{
    r->msg = msg;
    longjmp(r->fail, 1);
}

static void rd(struct reader *r, void *p, size_t n)
{
    if (fread(p, 1, n, r->f) != n)
        bad(r, feof(r->f) ? "unexpected end of file in the header" : strerror(errno));
}

static uint32_t u32(struct reader *r)
{
    uint32_t v;
    rd(r, &v, 4);
    return v;
}

static uint64_t u64(struct reader *r)
{
    uint64_t v;
    rd(r, &v, 8);
    return v;
}

/* A GGUF string: u64 length, bytes. Keeps at most cap-1 bytes in buf; returns the full length. */
static uint64_t str(struct reader *r, char *buf, size_t cap)
{
    uint64_t n = u64(r), keep = n < cap - 1 ? n : cap - 1;
    rd(r, buf, (size_t)keep);
    buf[keep] = 0;
    if (n > keep && fseeko(r->f, (off_t)(n - keep), SEEK_CUR) != 0)
        bad(r, "seek in a string");
    return n;
}

static size_t value_size(struct reader *r, uint32_t t)
{
    static const unsigned char sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t >= sizeof sz || sz[t] == 0)
        bad(r, "bad value type");
    return sz[t];
}

/* Reads one scalar of type t and prints it into out. */
static void scalar(struct reader *r, uint32_t t, char *out, size_t cap)
{
    unsigned char b[8];
    rd(r, b, value_size(r, t));
    switch (t) {
    case GGUF_U8: snprintf(out, cap, "%u", b[0]); break;
    case GGUF_I8: snprintf(out, cap, "%d", (int8_t)b[0]); break;
    case GGUF_BOOL: snprintf(out, cap, "%s", b[0] ? "true" : "false"); break;
    case GGUF_U16: { uint16_t v; memcpy(&v, b, 2); snprintf(out, cap, "%u", v); break; }
    case GGUF_I16: { int16_t v; memcpy(&v, b, 2); snprintf(out, cap, "%d", v); break; }
    case GGUF_U32: { uint32_t v; memcpy(&v, b, 4); snprintf(out, cap, "%" PRIu32, v); break; }
    case GGUF_I32: { int32_t v; memcpy(&v, b, 4); snprintf(out, cap, "%" PRId32, v); break; }
    case GGUF_F32: { float v; memcpy(&v, b, 4); snprintf(out, cap, "%.9g", (double)v); break; }
    case GGUF_U64: { uint64_t v; memcpy(&v, b, 8); snprintf(out, cap, "%" PRIu64, v); break; }
    case GGUF_I64: { int64_t v; memcpy(&v, b, 8); snprintf(out, cap, "%" PRId64, v); break; }
    case GGUF_F64: { double v; memcpy(&v, b, 8); snprintf(out, cap, "%.17g", v); break; }
    default: bad(r, "scalar of a non-scalar type");
    }
}

static void read_key(struct reader *r, struct gguf_key *k)
{
    char skip[8];
    str(r, k->name, sizeof k->name);
    k->type = u32(r);
    k->elem_type = 0;
    k->n = 1;
    k->text[0] = 0;
    if (k->type == GGUF_STR) {
        k->n = str(r, k->text, sizeof k->text);
    } else if (k->type == GGUF_ARR) {
        k->elem_type = u32(r);
        k->n = u64(r);
        if (k->elem_type == GGUF_STR) {
            for (uint64_t j = 0; j < k->n; j++)
                str(r, skip, sizeof skip);
        } else if (k->n <= 16) {
            for (uint64_t j = 0; j < k->n; j++) {
                char v[32];
                size_t len = strlen(k->text);
                scalar(r, k->elem_type, v, sizeof v);
                snprintf(k->text + len, sizeof k->text - len, "%s%s", j ? " " : "", v);
            }
        } else if (fseeko(r->f, (off_t)(k->n * value_size(r, k->elem_type)), SEEK_CUR) != 0) {
            bad(r, "seek over an array");
        }
    } else {
        scalar(r, k->type, k->text, sizeof k->text);
    }
}

static void read_tensor(struct reader *r, struct gguf_tensor *t)
{
    uint64_t ne = 1;
    str(r, t->name, sizeof t->name);
    t->n_dims = u32(r);
    if (t->n_dims > 4)
        bad(r, "tensor with more than 4 dimensions");
    for (uint32_t d = 0; d < t->n_dims; d++) {
        t->dim[d] = u64(r);
        ne *= t->dim[d];
    }
    t->type = u32(r);
    t->offset = u64(r); /* relative for now; made absolute once data_start is known */
    if (!gguf_tensor_type_name(t->type))
        bad(r, "tensor of an unknown ggml type");
    t->bytes = ne / gtypes[t->type].blck * gtypes[t->type].size;
}

int gguf_load(struct gguf *g, const char *path, char *err, size_t err_cap)
{
    struct reader r = {0};
    struct stat st;
    char magic[4];

    memset(g, 0, sizeof *g);
    r.f = fopen(path, "rb");
    if (!r.f || fstat(fileno(r.f), &st) != 0) {
        snprintf(err, err_cap, "%s: %s", path, strerror(errno));
        if (r.f)
            fclose(r.f);
        return -1;
    }
    if (setjmp(r.fail)) {
        snprintf(err, err_cap, "%s: %s", path, r.msg);
        fclose(r.f);
        gguf_free(g);
        return -1;
    }
    rd(&r, magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0)
        bad(&r, "not a GGUF file (magic)");
    g->version = u32(&r);
    g->n_tensors = u64(&r);
    g->n_keys = u64(&r);
    g->file_size = (uint64_t)st.st_size;
    g->keys = calloc(g->n_keys ? g->n_keys : 1, sizeof *g->keys);
    g->tensors = calloc(g->n_tensors ? g->n_tensors : 1, sizeof *g->tensors);
    if (!g->keys || !g->tensors)
        bad(&r, "out of memory");

    uint64_t alignment = 32, data_end = 0;
    for (uint64_t i = 0; i < g->n_keys; i++) {
        read_key(&r, &g->keys[i]);
        if (strcmp(g->keys[i].name, "general.alignment") == 0)
            alignment = strtoull(g->keys[i].text, NULL, 10);
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        struct gguf_tensor *t = &g->tensors[i];
        read_tensor(&r, t);
        g->data_bytes += t->bytes;
        if (t->offset + t->bytes > data_end)
            data_end = t->offset + t->bytes;
    }
    g->header_end = (uint64_t)ftello(r.f);
    g->data_start = (g->header_end + alignment - 1) / alignment * alignment;
    g->need = g->data_start + data_end;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        g->tensors[i].offset += g->data_start;
    fclose(r.f);
    return 0;
}

void gguf_free(struct gguf *g)
{
    free(g->keys);
    free(g->tensors);
    g->keys = NULL;
    g->tensors = NULL;
}

const struct gguf_key *gguf_key(const struct gguf *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_keys; i++)
        if (strcmp(g->keys[i].name, name) == 0)
            return &g->keys[i];
    return NULL;
}

const struct gguf_tensor *gguf_tensor(const struct gguf *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0)
            return &g->tensors[i];
    return NULL;
}

uint64_t gguf_key_u64(const struct gguf *g, const char *name, uint64_t fallback)
{
    const struct gguf_key *k = gguf_key(g, name);
    return k ? strtoull(k->text, NULL, 10) : fallback;
}
