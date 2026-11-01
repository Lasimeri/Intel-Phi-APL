#!/usr/bin/env bash
# doctor.sh: is this machine ready to build and run the APL engine? One
# PASS/FAIL/SKIP line per check with the value it saw; exit 0 only when
# every check passes (1 on any FAIL). Reads only: changes nothing.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

region=/dev/shm/phi-hostmem
region_bytes=4294967296 # HOSTMEM 4G in ~/.config/phi/cards

apl_stamp
echo

# Host side.
if v=$("$HOST_CC" --version 2>/dev/null | head -1); then pass "host cc: $v"; else fail "host cc '$HOST_CC' not found"; fi
if v=$(clangd --version 2>/dev/null | head -1); then pass "clangd: $v"; else fail "clangd not found (make compdb verifies with it)"; fi

# GNU APL, the host's F64 test reference: 2.0-1 and 2.0-2 are miscompiled by GCC 16
# (lifetime DSE: segfaults, corrupted values); tools/gnu-apl/PKGBUILD builds 2.0-3.
if v=$(pacman -Q gnu-apl 2>/dev/null) && [[ $v == 'gnu-apl 2.0-'[3-9]* ]]; then
    pass "GNU APL: $v"
elif [[ $APL_BIN != apl ]]; then
    pass "GNU APL: APL_BIN=$APL_BIN (installed: ${v:-none})"
else
    fail "GNU APL: ${v:-not installed}: install out/pkg/gnu-apl-2.0-3 (tools/gnu-apl builds it)"
fi

# The KNC vector compiler: present, and it accepts -march=knc with the sysroot.
if v=$("$PHI_LLVM_BIN/clang" --version 2>/dev/null | head -1); then
    if printf '#include <immintrin.h>\nint x;\n' | vpu_cc -x c - -fsyntax-only 2>/dev/null; then
        pass "KNC clang: $v (-march=knc, immintrin.h)"
    else
        fail "KNC clang $PHI_LLVM_BIN/clang rejects -march=knc or the sysroot $PHI_SYSROOT_7120A"
    fi
else
    fail "KNC clang not found at $PHI_LLVM_BIN/clang (set PHI_LLVM_BIN)"
fi

# The 7120A stack: knc-cc through toolchain/env.sh, the ISA auditor.
if v=$(card_cc --version 2>/dev/null | head -1); then pass "knc-cc: $v"; else fail "knc-cc not found via $PHI_TREE/toolchain/env.sh (set PHI_TREE)"; fi
if [[ -x "$PHI_TREE/host/target/debug/phi-isa-audit" ]]; then pass "phi-isa-audit: $PHI_TREE/host/target/debug/phi-isa-audit"; else fail "phi-isa-audit missing under $PHI_TREE/host/target/debug"; fi
if v=$(command -v phi); then pass "phi CLI: $v"; else fail "phi not on PATH (run phi install-cli from the 7120A stack)"; fi

# The card.
if card_up; then
    pass "card $PHI_CARD answers on the control socket"
    if sw=$(phic run cat /proc/swaps 2>/dev/null); then
        if /usr/bin/grep -q '/dev/phiblk1' <<<"$sw"; then
            fail "card swap is on /dev/phiblk1 = the whole shared region (HOSTMEM below 4G?)"
        elif /usr/bin/grep -q '/dev/loop0' <<<"$sw"; then
            pass "card swap on /dev/loop0 (region offset 2 GiB), [0, 2 GiB) free for parts"
        else
            pass "card has no swap on the shared region"
        fi
    else
        fail "cannot read /proc/swaps on the card"
    fi
else
    fail "card $PHI_CARD does not answer (phi -c $PHI_CARD up)"
fi

# The shared host-memory region.
if [[ -e $region ]]; then
    b=$(stat -c %s "$region")
    if ((b == region_bytes)); then pass "region $region: $b B"; else fail "region $region: $b B, want $region_bytes (HOSTMEM 4G)"; fi
else
    fail "region $region absent (card not up?)"
fi

# Models: present; complete by their own tensor tables when gguf_info is built.
while read -r _ name; do
    f="$apl_root/input/models/$name"
    if [[ ! -r $f ]]; then
        fail "model $name missing (see README: models)"
    elif [[ -x "$apl_out/host/gguf_info" ]]; then
        if r=$("$apl_out/host/gguf_info" "$f" 2>&1 | /usr/bin/grep -a -m1 -E 'COMPLETE|TRUNCATED'); then
            if [[ $r == *COMPLETE* ]]; then pass "model $name: $(stat -c %s "$f") B, $r"; else fail "model $name: $r"; fi
        else
            fail "model $name: gguf_info gave no verdict"
        fi
    else
        pass "model $name: $(stat -c %s "$f") B (run make host for the completeness check)"
    fi
done <"$apl_root/input/models/SHA256SUMS"

echo
verdict
