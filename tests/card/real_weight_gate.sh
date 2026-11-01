#!/usr/bin/env bash
# real_weight_gate.sh: a real weight matrix through the whole path: NVMe, region,
# card RAM, repack in place, W+.×x on the VPU. The input x is the APL reference's
# own last step, so y is the logits of the benchmark prompt.
#
#   real_weight_gate.sh [MODEL] [TPC CORES PF_L2 PF_L1 REPEATS]
#
# The gate, fixed before the first run (2026-11-01):
#   the file's sum on the card == the sum on the host;
#   card y == host-model y, bit for bit, every lane;
#   y within 1e-3 absolutely of the F64 APL reference's logits;
#   y against llama.cpp: tests/host/logit_cmp's gate (top-1, NRMSE, top 10).
# Exit 0 pass, 1 fail, 77 skip. Records go to out/card-gate/.
# shellcheck source=../../scripts/lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"

name=${1:-Qwen3.8-2B-Q8_0}
tpc=${2:-2} cores=${3:-59} pf_l2=${4:-32768} pf_l1=${5:-1024} repeats=${6:-15}
model="$apl_root/input/models/$name.gguf"
golden="$apl_out/apl/passing-2B"
w="$apl_out/card-gate"
tensor=token_embd.weight
card_dir=/tmp/apl
mkdir -p "$w"

card_up || { skip "real weights: the card does not answer"; verdict; }
[[ -r $golden/x-23.txt && -r $golden/logits.txt ]] ||
    { skip "real weights: no APL reference in $golden"; verdict; }

read -r offset cols rows < <("$apl_out/host/gguf_info" "$model" "$tensor")

# x: the output norm of the reference's last hidden state, computed in APL
"$apl_out/host/gguf_apl" "$model" >"$w/index.apl"
{
    cat "$w/index.apl" "$apl_root/src/apl/qwen35.apl" "$apl_root/src/apl/gguf_read.apl"
    cat <<APL
⎕PP←17
FH←⎕FIO[3] GGUF_PATH
W_ONORM←Weight 'output_norm.weight'
⊣⎕FIO[4] FH
X←(-N_EMBD)↑,⍎19 ⎕CR ⎕UCS ⎕FIO[26] '$golden/x-23.txt'
FH←'w' ⎕FIO[3] '$w/x.txt'
⊣(⍕W_ONORM RMS X) ⎕FIO[23] FH
⊣⎕FIO[4] FH
)OFF
APL
} >"$w/x.apl"
"$APL_BIN" --silent --noColor --noCIN -f "$w/x.apl" >"$w/x.log" 2>&1
if /usr/bin/grep -a -q ERROR "$w/x.log"; then fail "real weights: x in APL: see $w/x.log"; verdict; fi
"$apl_out/host/txt2f32" "$w/x.txt" "$w/x.f32" "$cols"

# the host model, one thread: the test reference for the card's bits
"$apl_out/host/real_gemv" "$model" "$offset" "$rows" "$cols" "$w/x.f32" "$w/y-host.f32" 1 1 0 0 1 >"$w/host.log" 2>&1 ||
    { fail "real weights: host model: see $w/host.log"; verdict; }

# the card: the feed starts first and waits for it
card_pool "$model"
phic put "$apl_out/card/real_gemv" "$card_dir/real_gemv" >/dev/null
phic put "$w/x.f32" "$card_dir/x.f32" >/dev/null
"$apl_out/host/apl_feed" "$model" >"$w/feed.log" 2>&1 &
feed=$!
if phic run "$card_dir/real_gemv" stream "$offset" "$rows" "$cols" "$card_dir/x.f32" "$card_dir/y.f32" "$tpc" "$cores" "$pf_l2" "$pf_l1" "$repeats" >"$w/card.log" 2>&1; then
    card_rc=0
else
    card_rc=1
fi
wait "$feed" && feed_rc=0 || feed_rc=$?
cat "$w/feed.log" "$w/card.log"
if ((card_rc || feed_rc)); then fail "real weights: card rc $card_rc, feed rc $feed_rc"; verdict; fi
phic get "$card_dir/y.f32" "$w/y-card.f32" >/dev/null

sum_host=$(/usr/bin/grep -a -o 'file sum [0-9a-f]*' "$w/host.log" || true)
sum_card=$(/usr/bin/grep -a -o 'file sum [0-9a-f]*' "$w/card.log" || true)
if [[ -n $sum_card && $sum_card == "$sum_host" ]]; then pass "the file on the card: $sum_card"; else fail "file sums: card [$sum_card], host [$sum_host]"; fi
if cmp -s "$w/y-card.f32" "$w/y-host.f32"; then pass "card y == host-model y, bit for bit: $((rows)) lanes"; else fail "card y differs from the host model's: $(cmp "$w/y-card.f32" "$w/y-host.f32" 2>&1 | cut -c1-120)"; fi
if v=$("$apl_out/host/numcmp" "$w/y-card.f32" "$golden/logits.txt" 0 1e-3); then pass "y vs F64 APL reference: $v"; else fail "y vs F64 APL reference: $v"; fi
if v=$("$apl_out/host/logit_cmp" "$w/y-card.f32" "$apl_out/ref/$name/logits.f32" 0 "$rows"); then pass "y vs llama.cpp: $(tail -1 <<<"$v")"; else fail "y vs llama.cpp: $v"; fi
verdict
