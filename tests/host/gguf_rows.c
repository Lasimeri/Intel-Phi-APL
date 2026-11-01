/* gguf_rows.c: print rows of a GGUF tensor as numbers, dequantized in plain
 * C (TEST ONLY): the independent check of the APL program's own Q8_0 and F32
 * decoding. Q8_0 uses the compiler's _Float16 conversion for the block scale.
 *
 *   gguf_rows FILE TENSOR FIRST_ROW N_ROWS
 *
 * One number per line, %.17g; a Q8_0 value (f16 scale x int8) and an F32
 * value are exact in double, so 17 digits round-trip. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf_reader.h"

int main(int argc, char **argv)
{
    struct gguf g;
    char err[512];

    if (argc != 5) {
        fprintf(stderr, "usage: gguf_rows FILE TENSOR FIRST_ROW N_ROWS\n");
        return 2;
    }
    if (gguf_load(&g, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "gguf_rows: %s\n", err);
        return 1;
    }
    const struct gguf_tensor *t = gguf_tensor(&g, argv[2]);
    if (!t || (t->type != GGML_Q8_0 && t->type != GGML_F32)) {
        fprintf(stderr, "gguf_rows: %s: no such Q8_0 or F32 tensor\n", argv[2]);
        return 1;
    }
    uint64_t first = strtoull(argv[3], NULL, 10), rows = strtoull(argv[4], NULL, 10), width = t->dim[0];
    uint64_t row_bytes = t->type == GGML_Q8_0 ? width / 32 * 34 : width * 4;
    FILE *f = fopen(argv[1], "rb");
    unsigned char *b = malloc((size_t)row_bytes);
    if (!f || !b || fseeko(f, (off_t)(t->offset + first * row_bytes), SEEK_SET) != 0)
        return 1;
    for (uint64_t r = 0; r < rows; r++) {
        if (fread(b, 1, (size_t)row_bytes, f) != row_bytes) {
            fprintf(stderr, "gguf_rows: short read\n");
            return 1;
        }
        if (t->type == GGML_F32) {
            for (uint64_t i = 0; i < width; i++) {
                float v;
                memcpy(&v, b + 4 * i, 4);
                printf("%.17g\n", (double)v);
            }
        } else {
            for (uint64_t k = 0; k < width / 32; k++) {
                _Float16 h;
                memcpy(&h, b + 34 * k, 2);
                double scale = (double)h;
                for (int i = 0; i < 32; i++)
                    printf("%.17g\n", scale * (double)(int8_t)b[34 * k + 2 + (uint64_t)i]);
            }
        }
    }
    free(b);
    fclose(f);
    gguf_free(&g);
    return 0;
}
