/* t_knc_half.c: host test of knc.h's binary16 -> binary32 decoder, the one
 * piece of the host model with its own arithmetic. All 65536 patterns
 * against the compiler's _Float16 conversion: every non-NaN pattern must
 * give the identical binary32, every NaN a NaN with the same sign and the
 * payload shifted into place (quiet bit aside: reported, not failed, since
 * the card's NaN handling is gate 2.1's question). Exit 0 on pass. */
#include <stdio.h>
#include <string.h>

#include "knc.h"

static uint32_t bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

int main(void)
{
    int bad = 0, nans = 0, quiet_diff = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        _Float16 f16;
        uint16_t hb = (uint16_t)h;
        memcpy(&f16, &hb, 2);
        uint32_t want = bits((float)f16), got = bits(knc_half(hb));
        int nan = ((h >> 10) & 31) == 31 && (h & 1023);
        if (nan) {
            nans++;
            quiet_diff += (got ^ want) == 0x00400000u;
            if (((got ^ want) & ~0x00400000u) != 0)
                bad++;
        } else if (got != want) {
            if (bad++ < 8)
                printf("h %04x got %08x want %08x\n", h, got, want);
        }
    }
    printf("knc_half: 65536 patterns, %d NaN (%d differ only in the quiet bit), %d wrong\n", nans, quiet_diff, bad);
    return bad != 0;
}
