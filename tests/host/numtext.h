/* numtext.h: read whitespace-separated numbers as text, understanding APL's high
 * minus (U+00AF) and E notation, so GNU APL output can be held against C output
 * (TEST ONLY). */
#ifndef APL_NUMTEXT_H
#define APL_NUMTEXT_H
#include <stdio.h>
#include <stdlib.h>

/* Reads the next number, or returns 0 at end of file. */
static inline int next_number(FILE *f, double *out)
{
    char tok[128];
    size_t n = 0;
    int c;
    while ((c = fgetc(f)) != EOF && (c == ' ' || c == '\n' || c == '\t' || c == '\r'))
        ;
    if (c == EOF)
        return 0;
    while (c != EOF && c != ' ' && c != '\n' && c != '\t' && c != '\r') {
        if (c == 0xC2) { /* first byte of the high minus */
            c = fgetc(f);
            if (c != 0xAF) {
                fprintf(stderr, "numtext: bad UTF-8 in a number\n");
                exit(2);
            }
            c = '-';
        }
        if (n + 1 < sizeof tok)
            tok[n++] = (char)c;
        c = fgetc(f);
    }
    tok[n] = 0;
    char *end;
    *out = strtod(tok, &end);
    if (*end) {
        fprintf(stderr, "numtext: not a number: %s\n", tok);
        exit(2);
    }
    return 1;
}

#endif
