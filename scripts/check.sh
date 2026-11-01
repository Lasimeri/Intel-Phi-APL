#!/usr/bin/env bash
# check.sh: every test of the APL engine, host first, then the card. One
# PASS/FAIL/SKIP line per check; exit 0 all passed, 1 any failed, 77 none
# failed but the card part was skipped (card down). The whole output, under
# a provenance stamp, is also kept as a run record in
# logs/gates/YYYY-MM-DD/HHMMSS-check.txt. Expects `make all` to have run.
#
# Golden files (tests/golden/) hold the expected output lines; they come
# from the independent references (fma_ref.c, upconv_cmp.c --ref), never
# from the card, so a card result is judged against arithmetic, not
# against itself.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

h="$apl_out/host" c="$apl_out/card" g="$apl_root/tests/golden" r="$apl_out/run"
mkdir -p "$r"
day=$(date +%F)
rec_dir="$apl_root/logs/gates/$day"
mkdir -p "$rec_dir"
rec="$rec_dir/$(date +%H%M%S)-check.txt"
exec > >(tee "$rec") 2>&1

# $1 label, $2 file with the produced lines, $3 golden file.
same() {
    if cmp -s "$2" "$3"; then
        pass "$1 == ${3#"$apl_root"/}"
    else
        fail "$1 differs from ${3#"$apl_root"/}"
        diff "$3" "$2" | head -10
    fi
}

apl_stamp
echo

echo "== host"
if "$h/t_knc_half" >"$r/t_knc_half.txt"; then pass "knc_half: $(cat "$r/t_knc_half.txt")"; else fail "knc_half: $(cat "$r/t_knc_half.txt")"; fi
while read -r _ name; do
    m="$apl_root/input/models/$name"
    if [[ ! -r $m ]]; then
        skip "gguf_info $name: model not present"
    elif v=$("$h/gguf_info" "$m" 2>&1 | /usr/bin/grep -a -m1 -E 'COMPLETE|TRUNCATED') && [[ $v == *COMPLETE* ]]; then
        pass "gguf_info $name: $v"
    else
        fail "gguf_info $name: ${v:-no verdict}"
    fi
done <"$apl_root/input/models/SHA256SUMS"
"$h/fma-gate" >"$r/gate-a.host.txt"
same "gate A host model" "$r/gate-a.host.txt" "$g/gate-a.txt"
"$h/fma-ref" >"$r/gate-a.ref.txt" 2>"$r/gate-a.ref.err"
same "gate A reference ($(cat "$r/gate-a.ref.err"))" "$r/gate-a.ref.txt" "$g/gate-a.txt"
"$h/upconv_cmp" --ref >"$r/gate-2.1.ref.txt"
same "gate 2.1 reference" "$r/gate-2.1.ref.txt" "$g/gate-2.1.txt"
"$h/upconv" "$r/upconv.host.bin" >"$r/gate-2.1.host.txt"
same "gate 2.1 host model" "$r/gate-2.1.host.txt" "$g/gate-2.1.txt"
if "$h/upconv_cmp" "$r/upconv.host.bin" >"$r/gate-2.1.host-cmp.txt"; then
    pass "gate 2.1 host model == reference, every lane ($(head -1 "$r/gate-2.1.host-cmp.txt"))"
else
    fail "gate 2.1 host model vs reference"
    cat "$r/gate-2.1.host-cmp.txt"
fi

if v=$("$apl_root/tests/apl/parse_oracle.sh" 2>&1); then pass "APL parser oracle: $v"; else fail "APL parser oracle: $v"; fi
if v=$("$apl_root/tests/apl/shape_oracle.sh" 2>&1); then pass "APL shape oracle: $v"; else fail "APL shape oracle: $v"; fi
if v=$("$apl_root/tests/apl/batch_oracle.sh" 2>&1); then pass "APL batch oracle: $v"; else fail "APL batch oracle: $v"; fi
if v=$("$apl_root/tests/apl/delta_witness.sh" 2>&1); then pass "APL delta witness: $v"; else fail "APL delta witness: $v"; fi
echo
echo "== card $PHI_CARD"
if ! card_up; then
    skip "card $PHI_CARD does not answer: gate A and gate 2.1 on the card not run"
    echo
    verdict
fi
phic run mkdir -p "$apl_card_dir"
for b in fma-gate upconv gemv_bench; do
    phic put "$c/$b" "$apl_card_dir/$b" >/dev/null
done

if phic run "$apl_card_dir/fma-gate" >"$r/gate-a.card.txt"; then
    same "gate A card" "$r/gate-a.card.txt" "$g/gate-a.txt"
else
    fail "gate A card: fma-gate exited $?"
fi

if phic run "$apl_card_dir/upconv" "$apl_card_dir/upconv.bin" >"$r/gate-2.1.card.txt"; then
    same "gate 2.1 card" "$r/gate-2.1.card.txt" "$g/gate-2.1.txt"
    phic get "$apl_card_dir/upconv.bin" "$r/upconv.card.bin" >/dev/null
    if "$h/upconv_cmp" "$r/upconv.host.bin" "$r/upconv.card.bin" >"$r/gate-2.1.card-cmp.txt"; then
        pass "gate 2.1 card == reference, every lane"
    else
        fail "gate 2.1 card vs reference (lane detail below)"
    fi
    cat "$r/gate-2.1.card-cmp.txt"
else
    fail "gate 2.1 card: upconv exited $?"
fi

# Alignment probes. Measured 2026-11-01 (logs/gates/2026-11-01/055249-check.txt):
# an operand at its own alignment runs (exit 0); the same operand 1 byte
# off (2 for float16) faults with SIGSEGV. Both halves are asserted, so a
# change of kernel code (say, to an unaligned loadunpack) shows up here.
if phic run "$apl_card_dir/upconv" --probe >"$r/gate-2.1.probe.txt"; then
    while read -r line; do
        case $line in
        *"+16 exit 0"* | *"+32 exit 0"* | *"+4  exit 0"*) pass "$line" ;;
        *"+16 "* | *"+32 "* | *"+4  "*) fail "$line (aligned operand must run)" ;;
        *"+1  signal 11 "* | *"+2  signal 11 "*) pass "$line" ;;
        *) fail "$line (misaligned operand: SIGSEGV expected)" ;;
        esac
    done <"$r/gate-2.1.probe.txt"
else
    fail "gate 2.1 probes: upconv --probe exited $?"
    cat "$r/gate-2.1.probe.txt"
fi

echo
echo "== real weights and the whole engine, card $PHI_CARD with the host's worker"
run_gate() {
    local out rc=0
    out=$("$@" 2>&1) || rc=$?
    printf '%s\n' "$out" | /usr/bin/grep -a -E '^(PASS|FAIL|SKIP|DECODE)' >>"$r/engine-gates.txt" || true
    while read -r line; do
        case $line in
        PASS*) pass "${line#PASS  }" ;;
        FAIL*) fail "${line#FAIL  }" ;;
        SKIP*) skip "${line#SKIP  }" ;;
        DECODE*) echo "      $line" ;;
        esac
    done < <(printf '%s\n' "$out" | /usr/bin/grep -a -E '^(PASS|FAIL|SKIP|DECODE)')
    ((rc == 0 || rc == 77)) || fail "$1 exited $rc"
}
# X+.×⍉W for T tokens against one matvec per token, bit for bit: T crosses every chunk size
gemm_bad=0
gemm_lanes=0
for T in 1 2 3 4 5 6 7 8 9 15 16 17 40; do
    v=$(phic run "$apl_card_dir/gemv_bench" gemm 1888 512 "$T" 2 59 4096 2048 3 2>&1 | /usr/bin/grep -a '^gemm' || true)
    hv=$("$h/gemv_bench" gemm 16 512 "$T" 1 1 0 0 1 2>&1 | /usr/bin/grep -a "^gemm" || true)
    [[ $v == *"lanes differing 0 of "* && $hv == *"lanes differing 0 of "* ]] || gemm_bad=$((gemm_bad + 1))
    gemm_lanes=$((gemm_lanes + 1888 * T + 16 * T))
done
# fewer groups than threads: the split by tokens as well
for R in 16 64 512; do
    for T in 2 8 16; do
        v=$(phic run "$apl_card_dir/gemv_bench" gemm "$R" 512 "$T" 2 59 4096 2048 3 2>&1 | /usr/bin/grep -a '^gemm' || true)
        [[ $v == *"lanes differing 0 of "* ]] || gemm_bad=$((gemm_bad + 1))
        gemm_lanes=$((gemm_lanes + R * T))
    done
done
if ((gemm_bad == 0)); then pass "gemm == one gemv per token, bit for bit, card and host model: $gemm_lanes lanes, 22 runs"; else fail "gemm against gemv: $gemm_bad of 22 runs differ"; fi

run_gate "$apl_root/tests/card/real_weight_gate.sh"
run_gate "$apl_root/tests/card/engine_gate.sh"

echo
echo "record: ${rec#"$apl_root"/}"
verdict
