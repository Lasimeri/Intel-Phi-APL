# lib.sh: sourced by every script of the APL engine (bash). One place for
# the paths, the three compilers and their flags, the card verbs, the exit
# codes, and the provenance stamp every run record starts with.
#
# Environment overrides (all optional; config.mk passes make's values in):
#   PHI_TREE           the Intel Phi 7120A stack (toolchain/env.sh, phi-isa-audit)
#   PHI_LLVM_BIN       the Intel-Phi-LLVM KNC clang bin directory (-march=knc)
#   PHI_SYSROOT_7120A  the card musl sysroot the VPU objects compile against
#   PHI_CARD           card index for phi -c (default 0)
#   HOST_CC            host C compiler (default cc)
#   LLAMA_DIR          an existing llama.cpp build, read only: the reference (test only)
#   APL_BIN            the GNU APL interpreter (default apl)
set -euo pipefail

apl_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PHI_TREE:=/media/1TB-NVMe/Xeon-Phi/Intel Phi 7120A}"
: "${PHI_LLVM_BIN:=$HOME/.cache/intel-phi-llvm-build/llvm/bin}"
: "${PHI_SYSROOT_7120A:=$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
: "${PHI_CARD:=0}"
: "${HOST_CC:=cc}"
: "${LLAMA_DIR:=$HOME/llama.cpp}"
: "${APL_BIN:=apl}"

# Exit codes of check-style scripts: automake's convention.
APL_PASS=0
APL_FAIL=1
APL_SKIP=77

apl_out="$apl_root/out"
apl_card_dir=/tmp/apl # on the card (tmpfs)
# The OpenCL C header, for the offline weight tools (src/gpu); not in /usr/include here
apl_cl_include=${APL_CL_INCLUDE:-/opt/cuda/targets/x86_64-linux/include}

# Host builds: no -march (x86-64 baseline, no FMA unit in play) and no FP
# contraction, so a*b+c in the knc.h host model stays two roundings unless
# the source asks for fmaf.
host_flags=(-std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -I "$apl_root/src/card" -I "$apl_root/src/gate" -I "$apl_root/src/host")
# VPU objects: the KNC clang, kernels only (pointers and integers at the
# boundary, never a v16f by value: the card mains use the x87 ABI).
vpu_flags=(--target=x86_64-unknown-linux-musl -march=knc "--sysroot=$PHI_SYSROOT_7120A" -std=gnu11 -O2
           -Wall -Wextra -Werror -ffp-contract=off -fno-jump-tables -falign-loops=1 -fPIE
           -I "$apl_root/src/card" -I "$apl_root/src/gate")
# Card mains: the stack's knc-cc (x87 ABI, static musl). It does not
# predefine __KNC__, so a main must never include knc.h.
card_flags=(-std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -static -I "$apl_root/src/gate" -I "$apl_root/src/card")

host_cc() { "$HOST_CC" "${host_flags[@]}" "$@"; }
vpu_cc() { "$PHI_LLVM_BIN/clang" "${vpu_flags[@]}" "$@"; }
card_cc() { (
    # shellcheck disable=SC1091
    . "$PHI_TREE/toolchain/env.sh"
    knc-cc "${card_flags[@]}" "$@"
); }
isa_audit() { "$PHI_TREE/host/target/debug/phi-isa-audit" "$@"; }
phic() { phi -c "$PHI_CARD" "$@"; }

# The card's huge pages for a model file: its pages + a margin for state, weights and
# frame, on top of what is in use (vgd's stay). Back to the earlier total on exit.
card_pool() {
    local total used need
    read -r total used < <(phic run awk '/HugePages_Total/ { t = $2 } /HugePages_Free/ { f = $2 } END { print t, t - f }' /proc/meminfo)
    need=$((($(stat -c %s "$1") >> 21) + 150))
    ((need + used > total)) || return 0
    phic run sh -c "echo $((need + used)) > /proc/sys/vm/nr_hugepages" >/dev/null
    # shellcheck disable=SC2064
    trap "phic run sh -c 'echo $total > /proc/sys/vm/nr_hugepages' >/dev/null" EXIT
}

# The card answers a command through the control socket.
card_up() { phic run true >/dev/null 2>&1; }

# Audit a card binary; fail unless 0 illegal and 0 suspect. Prints the
# result line.
audit_clean() {
    local line
    line=$(isa_audit "$1" | /usr/bin/grep -a '^result:')
    printf '%s: %s\n' "${1##*/}" "$line"
    [[ $line == "result: 0 illegal, 0 suspect,"* ]]
}

# Provenance: who built and ran what, where, when.
apl_stamp() {
    local rev stack st bdf
    rev=$(git -C "$apl_root" describe --always --dirty 2>/dev/null || echo "no commits")
    stack=$(git -C "$PHI_TREE" describe --always --dirty 2>/dev/null || echo "unknown")
    # Capture first, parse after: grep -m1 in a pipe would SIGPIPE phi
    # status, and pipefail would turn a found address into a failure.
    st=$(phic status 2>/dev/null || true)
    bdf=$(/usr/bin/grep -a -o -m1 -E '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]' <<<"$st" || true)
    [[ -n $bdf ]] || bdf="card unreachable"
    printf 'date     %s\n' "$(date '+%F %T %Z')"
    printf 'engine   %s (git %s)\n' "$apl_root" "$rev"
    printf 'stack    %s (git %s)\n' "$PHI_TREE" "$stack"
    printf 'vpu cc   %s\n' "$("$PHI_LLVM_BIN/clang" --version 2>/dev/null | head -1)"
    printf 'host cc  %s\n' "$("$HOST_CC" --version 2>/dev/null | head -1)"
    printf 'card     %s %s\n' "$PHI_CARD" "$bdf"
}

# PASS/FAIL/SKIP lines, counted.
apl_n_pass=0 apl_n_fail=0 apl_n_skip=0
pass() { apl_n_pass=$((apl_n_pass + 1)); printf 'PASS  %s\n' "$*"; }
fail() { apl_n_fail=$((apl_n_fail + 1)); printf 'FAIL  %s\n' "$*"; }
skip() { apl_n_skip=$((apl_n_skip + 1)); printf 'SKIP  %s\n' "$*"; }
# Exit with the verdict of the counted lines: FAIL beats SKIP beats PASS.
verdict() {
    printf -- '-- %d passed, %d failed, %d skipped\n' "$apl_n_pass" "$apl_n_fail" "$apl_n_skip"
    ((apl_n_fail == 0)) || exit "$APL_FAIL"
    ((apl_n_skip == 0)) || exit "$APL_SKIP"
    exit "$APL_PASS"
}

# The fixed benchmark and verification prompt (real text: the opening of the
# English Wikipedia article on APL), shared by the llama.cpp reference and
# the card runs so both see the same token IDs.
APL_PROMPT="APL (named after the book A Programming Language) is a programming language developed in the 1960s by Kenneth E. Iverson. Its central datatype is the multidimensional array."
