#!/usr/bin/env bash
# batch_oracle.sh: does a batch of NB tokens mean the same as NB steps of one? qwen35.apl runs
# on the small synthetic model in GNU APL (F64): the same tokens as one call of NB=3, as 2 + 1,
# and one by one, each from zero state. Layer outputs, states and logits must agree within
# 1e-12 absolutely. Exit 0 pass, 1 fail. Prints one summary line.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"
w="$apl_out/apl/batch-oracle"
mkdir -p "$w"
{
    printf '⎕IO←0\n⎕CT←0\n⎕PP←17\n⎕PW←1000\n'
    cat "$apl_root/src/apl/qwen35.apl"
    sed -n '/^D←32$/,/^X←Rnd 3,D$/p' "$apl_root/tests/apl/small_model.apl"
    cat <<'APL'
∇Reset
 WIN←(WL,NK+NK+NV)⍴0
 S←(SL,HV,ST,ST)⍴0
 KC←(CTX,HKV,HD)⍴0
 VC←KC
∇
⍝ Each layer's output through a schedule of batch sizes, then the state: one long vector
∇Z←(F RunSizes) SIZES;P;K;R
 Reset
 Z←⍳0
 P←K←0
NEXT:→(K≥⍴SIZES)/DONE
 NB←SIZES[K]
 R←P F X[P+⍳NB;]
 Z←Z,,R
 P←P+NB
 K←K+1
 →NEXT
DONE:Z←Z,(,S),(,WIN),(,KC),,VC
∇
⎕←'delta 3 vs 1 1 1' (⌈/|(DeltaLayer RunSizes ,3)-DeltaLayer RunSizes 1 1 1)
⎕←'delta 2 1 vs 1 1 1' (⌈/|(DeltaLayer RunSizes 2 1)-DeltaLayer RunSizes 1 1 1)
⎕←'attn 3 vs 1 1 1' (⌈/|(AttnLayer RunSizes ,3)-AttnLayer RunSizes 1 1 1)
⎕←'attn 1 2 vs 1 1 1' (⌈/|(AttnLayer RunSizes 1 2)-AttnLayer RunSizes 1 1 1)
)OFF
APL
} >"$w/run.apl"
"$APL_BIN" --silent --noColor --noCIN -f "$w/run.apl" >"$w/run.out" 2>&1
/usr/bin/grep -a -q 'ERROR' "$w/run.out" && { echo "batch oracle errored (see $w/run.out)"; exit 1; }
awk 'NF { n++; v = $NF + 0; a = v < 0 ? -v : v; if (a > w) w = a; if (a > 1e-12) bad++ } END { printf "batch == steps: %d schedules, worst %.3e (limit 1e-12)\n", n, w; exit (bad > 0 || n != 4) }' "$w/run.out"
