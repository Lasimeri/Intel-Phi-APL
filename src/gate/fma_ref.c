/* fma_ref.c: gate A, the independent reference (host cc, -lm). Same inputs
 * as fma_main.c, c = fmaf(a, b, c) from glibc (correctly rounded: one
 * rounding) in plain scalar code that shares nothing with knc.h, and the
 * same output line. On stderr: how many lanes an unfused a*b then +c (two
 * binary32 roundings) would change, i.e. how many lanes prove that the
 * card rounds once (809 of 4096 on 2026-10-31). */
#include <math.h>
#include <stdio.h>

#include "gate_common.h"

static float A[GATE_N], B[GATE_N], C[GATE_N];

int main(void)
{
    int unfused_diff = 0;
    gate_fill(A, B, C, GATE_N);
    for (int i = 0; i < GATE_N; i++) {
        volatile float p = A[i] * B[i];
        volatile float u = p + C[i];
        float f = fmaf(A[i], B[i], C[i]);
        unfused_diff += gate_bits(u) != gate_bits(f);
        C[i] = f;
    }
    printf("n %d hash %016llx c0 %08x c1 %08x clast %08x\n", GATE_N,
           (unsigned long long)gate_hash(C, GATE_N), (unsigned)gate_bits(C[0]),
           (unsigned)gate_bits(C[1]), (unsigned)gate_bits(C[GATE_N - 1]));
    fprintf(stderr, "lanes where unfused mul+add differs from fmaf: %d of %d\n", unfused_diff, GATE_N);
    return 0;
}
