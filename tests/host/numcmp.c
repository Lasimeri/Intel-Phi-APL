/* numcmp.c: compare two number files (TEST ONLY). A file whose name ends in .f32 is
 * raw F32; any other is whitespace-separated text, APL's high minus and E notation
 * understood, so GNU APL output can be held against C output.
 *
 *   numcmp A B [REL_TOL [ABS_TOL]]     default REL_TOL 1e-15, ABS_TOL 0
 *
 * Prints the count and the largest absolute and relative difference; exit 0
 * when the counts match and every value is within REL_TOL (relative to the
 * larger magnitude; two exact zeros are equal) or within ABS_TOL absolutely (a
 * relative test alone fails on near-zero values, where round-off is all there
 * is), 1 otherwise. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "numtext.h"

struct source {
    FILE *f;
    int raw;
};

static struct source open_source(const char *path)
{
    const size_t n = strlen(path);
    struct source s = {.f = fopen(path, "r"), .raw = n > 4 && strcmp(path + n - 4, ".f32") == 0};
    if (!s.f) {
        perror(path);
        exit(2);
    }
    return s;
}

static int next_value(const struct source *s, double *out)
{
    if (!s->raw)
        return next_number(s->f, out);
    float v;
    if (fread(&v, sizeof v, 1, s->f) != 1)
        return 0;
    *out = (double)v;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: numcmp A B [REL_TOL [ABS_TOL]]\n");
        return 2;
    }
    double tol = argc >= 4 ? atof(argv[3]) : 1e-15, abs_tol = argc == 5 ? atof(argv[4]) : 0, max_abs = 0, max_rel = 0;
    const struct source a = open_source(argv[1]), b = open_source(argv[2]);
    long n = 0, over = 0;
    for (;;) {
        double x, y;
        int ha = next_value(&a, &x), hb = next_value(&b, &y);
        if (ha != hb) {
            printf("numcmp: different counts: one file ends after %ld numbers\n", n);
            return 1;
        }
        if (!ha)
            break;
        double d = fabs(x - y), m = fmax(fabs(x), fabs(y)), r = m > 0 ? d / m : 0;
        if (d > max_abs)
            max_abs = d;
        if (r > max_rel)
            max_rel = r;
        over += r > tol && d > abs_tol;
        n++;
    }
    printf("numcmp: %ld numbers, max abs diff %.3e, max rel diff %.3e, %ld beyond %.1e rel and %.1e abs\n", n, max_abs, max_rel, over, tol, abs_tol);
    return over != 0;
}
