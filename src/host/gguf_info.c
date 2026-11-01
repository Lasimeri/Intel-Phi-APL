/* gguf_info.c: report what the APL engine needs to choose and stream a
 * model: every metadata key, the bytes per ggml type, the tensors of layer 0
 * and outside the layers, and whether the file holds all the data its tensor
 * table names (the end computed from the table against the file size).
 * Only the header is read; the tensor data is never touched.
 * Usage: gguf_info FILE [-a]   (-a: list every tensor)
 * Exit 0 complete, 3 truncated, 1 not readable, 2 usage. */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "gguf_reader.h"

static void print_keys(const struct gguf *g)
{
    for (uint64_t i = 0; i < g->n_keys; i++) {
        const struct gguf_key *k = &g->keys[i];
        if (k->type == GGUF_STR) {
            printf("  %-48s str  %s%s\n", k->name, k->text, k->n >= sizeof k->text ? "..." : "");
        } else if (k->type == GGUF_ARR) {
            printf("  %-48s arr  %s[%" PRIu64 "]%s%s\n", k->name, gguf_value_type_name(k->elem_type), k->n,
                   k->text[0] ? "  " : "", k->text);
        } else {
            printf("  %-48s %-4s %s\n", k->name, gguf_value_type_name(k->type), k->text);
        }
    }
}

static void print_tensor(const struct gguf_tensor *t)
{
    char dims[96] = "";
    for (uint32_t d = 0; d < t->n_dims; d++)
        snprintf(dims + strlen(dims), sizeof dims - strlen(dims), "%s%" PRIu64, d ? " x " : "", t->dim[d]);
    printf("  %-40s %-8s %-28s %12" PRIu64 " B\n", t->name, gguf_tensor_type_name(t->type), dims, t->bytes);
}

int main(int argc, char **argv)
{
    struct gguf g;
    char err[512];
    uint64_t type_bytes[GGUF_MAX_TENSOR_TYPE] = {0}, type_count[GGUF_MAX_TENSOR_TYPE] = {0};
    int all = argc > 2 && strcmp(argv[2], "-a") == 0;

    if (argc < 2) {
        fprintf(stderr, "usage: gguf_info FILE [-a | TENSOR]\n");
        return 2;
    }
    if (gguf_load(&g, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "gguf_info: %s\n", err);
        return 1;
    }
    /* one tensor, for scripts: absolute offset, then ne0, ne1 */
    if (argc > 2 && !all) {
        const struct gguf_tensor *t = gguf_tensor(&g, argv[2]);
        if (!t) {
            fprintf(stderr, "gguf_info: no tensor %s\n", argv[2]);
            return 1;
        }
        printf("%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", t->offset, t->dim[0], t->dim[1]);
        return 0;
    }
    printf("file %s\nsize %" PRIu64 " B\nversion %" PRIu32 "  tensors %" PRIu64 "  keys %" PRIu64 "\n", argv[1], g.file_size,
           g.version, g.n_tensors, g.n_keys);
    print_keys(&g);

    printf("tensors (layer 0 and outside the layers%s):\n", all ? ", all" : "");
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        const struct gguf_tensor *t = &g.tensors[i];
        type_bytes[t->type] += t->bytes;
        type_count[t->type]++;
        if (all || strncmp(t->name, "blk.", 4) != 0 || strncmp(t->name, "blk.0.", 6) == 0)
            print_tensor(t);
    }
    printf("bytes by type:\n");
    for (uint32_t t = 0; t < GGUF_MAX_TENSOR_TYPE; t++)
        if (type_count[t])
            printf("  %-8s %5" PRIu64 " tensors %14" PRIu64 " B\n", gguf_tensor_type_name(t), type_count[t], type_bytes[t]);
    printf("header %" PRIu64 " B, data start %" PRIu64 ", tensor data %" PRIu64 " B, needed file size %" PRIu64 " B\n", g.header_end,
           g.data_start, g.data_bytes, g.need);
    int complete = g.file_size >= g.need;
    if (complete)
        printf("COMPLETE: file holds every tensor (%" PRIu64 " B spare)\n", g.file_size - g.need);
    else
        printf("TRUNCATED: file is %" PRIu64 " B short of its tensor table\n", g.need - g.file_size);
    gguf_free(&g);
    return complete ? 0 : 3;
}
