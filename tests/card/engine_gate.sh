#!/usr/bin/env bash
# engine_gate.sh: the whole engine: qwen35.apl through the compiler, the card's half on
# the VPU and the host's half on the worker, on a real model.
#
#   engine_gate.sh [MODEL] [HOST_THREADS]
#
# The gate, fixed before the first run (2026-11-01), the prompt in its largest batches:
#   every layer's output [T D] within 1e-2 absolutely of the F64 APL reference;
#   the logits within NRMSE 1e-3 of it, same top-1;
#   against llama.cpp: tests/host/logit_cmp's gate, and every generated token the same;
#   every repeat generates the same tokens.
# Then the prompt in batches of at most 4, 2 and 1: every compiled size, each layer within
# the same 1e-2 of the reference and the same next token.
# Exit 0 pass, 1 fail, 77 skip. The reference exists for the 2B (make baseline, apl_ref.sh).
# shellcheck source=../../scripts/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"

name=${1:-Qwen3.8-2B-Q8_0}
threads=${2:-4}
golden="$apl_out/apl/passing-2B"
gen="$apl_out/gen/$name"
dump="$gen/dump-gate"
n_gen=128

card_up || { skip "engine: the card does not answer"; verdict; }
[[ -r $golden/x-23.txt && -r $golden/logits.txt && -r $apl_out/ref/$name/logits.f32 ]] ||
    { skip "engine: no reference in $golden and out/ref/$name"; verdict; }

"$apl_root/scripts/engine.sh" compile "$name" >/dev/null || { fail "engine: the compile"; verdict; }
"$apl_root/scripts/engine.sh" build "$name" >/dev/null || { fail "engine: the build"; verdict; }
rm -rf "$dump"
log="$gen/gate.log"
if ! "$apl_root/scripts/engine.sh" split "$name" "$n_gen" 2 59 "$threads" 3 "$dump" >"$log" 2>&1; then
    fail "engine: the run: see $log"
    verdict
fi

bad=0 worst=0 n=0
while read -r f; do
    b=${f##*/}
    b=${b%.f32}
    if ! v=$("$apl_out/host/numcmp" "$f" "$golden/$b.txt" 0 1e-2); then bad=$((bad + 1)); fi
    worst=$(awk -v a="$worst" -v b="$(awk '{ print $7 + 0 }' <<<"$v")" 'BEGIN { print (b > a) ? b : a }')
    n=$((n + 1))
done < <(find "$dump" -name 'x-*.f32' | LC_ALL=C sort)
if ((n > 0 && bad == 0)); then pass "every layer within 1e-2 of the F64 reference: $n layers, worst $worst"; else fail "$bad of $n layers beyond 1e-2 (worst $worst)"; fi

"$apl_out/host/txt2f32" "$golden/logits.txt" "$gen/ref-logits.f32" "$(($(stat -c %s "$dump/logits.f32") / 4))"
v=$("$apl_out/host/logit_cmp" "$dump/logits.f32" "$gen/ref-logits.f32" 0 "$(($(stat -c %s "$dump/logits.f32") / 4))" || true)
nrmse=$(awk '/^error:/ { for (i = 1; i <= NF; i++) if ($i == "NRMSE") print $(i + 1) + 0 }' <<<"$v")
if awk -v x="$nrmse" 'BEGIN { exit !(x <= 1e-3) }'; then pass "logits against the F64 reference: NRMSE $nrmse (limit 1e-3)"; else fail "logits against the F64 reference: NRMSE $nrmse"; fi

if v=$("$apl_out/host/logit_cmp" "$dump/logits.f32" "$apl_out/ref/$name/logits.f32" 0 "$(($(stat -c %s "$dump/logits.f32") / 4))"); then pass "logits against llama.cpp: $(tail -1 <<<"$v")"; else fail "logits against llama.cpp: $v"; fi

same=$(awk 'FNR == 1 { f++ } /^gen / { if (f == 1) { for (i = 2; i <= NF; i++) a[i] = $i; n = NF } else { for (i = 2; i <= NF && i <= n; i++) s += ($i == a[i]); printf "%d %d", s + 0, n - 1 } }' \
    "$apl_out/ref/$name/tokens.txt" "$log")
if [[ ${same%% *} == "${same##* }" && -n $same ]]; then pass "generated tokens against llama.cpp: ${same%% *} of ${same##* } the same"; else fail "generated tokens against llama.cpp: $same"; fi

if /usr/bin/grep -a -q 'same tokens: yes' "$log"; then pass "every repeat generated the same tokens"; else fail "the repeats differ"; fi
echo "$(/usr/bin/grep -a '^DECODE' "$log" | cut -c1-120)"

# Speculation may change the speed, never the tokens: K=1 is the default run above
for k in 0 2; do
    log3="$gen/gate-k$k.log"
    if ! APL_MTP=$k "$apl_root/scripts/engine.sh" split "$name" "$n_gen" 2 59 "$threads" 1 >"$log3" 2>&1; then
        fail "engine: the run at K=$k: see $log3"
        continue
    fi
    same=$(awk 'FNR == 1 { f++ } /^gen / { if (f == 1) { for (i = 2; i <= NF; i++) a[i] = $i; n = NF } else if (!d) { for (i = 2; i <= NF; i++) if ($i == a[i]) c++; d = 1 } } END { print c + 0, n - 1 }' "$apl_out/ref/$name/tokens.txt" "$log3")
    if [[ ${same%% *} == "${same##* }" && ${same%% *} -gt 0 ]]; then pass "tokens at K=$k against llama.cpp: ${same%% *} of ${same##* }"; else fail "tokens at K=$k against llama.cpp: $same"; fi
done

for limit in 4 2 1; do
    d="$gen/dump-gate-$limit"
    rm -rf "$d"
    log2="$gen/gate-$limit.log"
    if ! APL_PROMPT_NB=$limit "$apl_root/scripts/engine.sh" split "$name" 2 2 59 "$threads" 1 "$d" >"$log2" 2>&1; then
        fail "engine: the run in batches of at most $limit: see $log2"
        continue
    fi
    bad=0 n=0 worst=0
    while read -r f; do
        b=${f##*/}
        b=${b%.f32}
        if ! v=$("$apl_out/host/numcmp" "$f" "$golden/$b.txt" 0 1e-2); then bad=$((bad + 1)); fi
        worst=$(awk -v a="$worst" -v b="$(awk '{ print $7 + 0 }' <<<"$v")" 'BEGIN { print (b > a) ? b : a }')
        n=$((n + 1))
    done < <(find "$d" -name 'x-*.f32' | LC_ALL=C sort)
    first=$(awk '/^gen / { print $2; exit }' "$log")
    again=$(awk '/^gen / { print $2; exit }' "$log2")
    if ((n > 0 && bad == 0)) && [[ $first == "$again" ]]; then pass "prompt in batches of at most $limit: $n layers, worst $worst, next token $again the same"; else fail "prompt in batches of at most $limit: $bad of $n layers beyond 1e-2, next token $again against $first"; fi
done
verdict
