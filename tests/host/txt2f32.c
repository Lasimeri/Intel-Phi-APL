/* txt2f32.c: whitespace-separated numbers as text (APL's high minus understood) to
 * raw F32, the last N of them (TEST ONLY): a reference's values as a kernel's input.
 *
 *   txt2f32 IN OUT N     exit 0, or 1 if IN holds fewer than N numbers */
#include <stdio.h>
#include <stdlib.h>

#include "numtext.h"

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: txt2f32 IN OUT N\n");
        return 2;
    }
    const long want = atol(argv[3]);
    FILE *in = fopen(argv[1], "r");
    float *ring = malloc(sizeof(float) * (size_t)want);
    if (!in || !ring) {
        perror("txt2f32");
        return 2;
    }
    long seen = 0;
    double v;
    while (next_number(in, &v))
        ring[seen++ % want] = (float)v;
    fclose(in);
    if (seen < want) {
        fprintf(stderr, "txt2f32: %s holds %ld numbers, wanted %ld\n", argv[1], seen, want);
        return 1;
    }
    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        perror(argv[2]);
        return 2;
    }
    for (long i = 0; i < want; i++)
        fwrite(&ring[(seen + i) % want], sizeof(float), 1, out);
    if (fclose(out)) {
        perror(argv[2]);
        return 2;
    }
    return 0;
}
