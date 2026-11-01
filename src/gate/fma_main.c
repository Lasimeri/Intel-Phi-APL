/* fma_main.c: gate A, the program. Built twice: by knc-cc against the card
 * kernel (runs on the card) and by the host cc against the host model of
 * the same kernel. Builds the inputs of gate_common.h, runs apl_gate_fma16
 * once, prints one line: the result hash and three result words. Both
 * builds, the independent reference fma_ref.c and tests/golden/gate-a.txt
 * must print the identical line. Never includes knc.h (x87 ABI side). */
#include <stdio.h>

#include "gate_common.h"
#include "gate_kernels.h"

static float A[GATE_N] __attribute__((aligned(64)));
static float B[GATE_N] __attribute__((aligned(64)));
static float C[GATE_N] __attribute__((aligned(64)));

int main(void)
{
    gate_fill(A, B, C, GATE_N);
    apl_gate_fma16(A, B, C, GATE_N);
    printf("n %d hash %016llx c0 %08x c1 %08x clast %08x\n", GATE_N,
           (unsigned long long)gate_hash(C, GATE_N), (unsigned)gate_bits(C[0]),
           (unsigned)gate_bits(C[1]), (unsigned)gate_bits(C[GATE_N - 1]));
    return 0;
}
