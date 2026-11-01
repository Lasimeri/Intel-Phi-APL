#!/usr/bin/env bash
# apl_ref.sh: run the APL source of the model (src/apl/qwen35.apl) in GNU APL on
# the host: the engine's F64 TEST REFERENCE, never an inference path.
#
#   apl_ref.sh dq MODEL                  check APL's dequantization against C
#   apl_ref.sh run MODEL [LAST_LAYER]    forward pass over the benchmark prompt
#   apl_ref.sh cmp MODEL GOLDEN_DIR      the run's dumps against a golden run,
#                                        each number within 1e-12 absolutely
#
# MODEL is a file name in input/models. run takes the prompt's token IDs from
# the llama.cpp reference (out/ref/MODEL/tokens.txt, made by make baseline),
# leaves one text dump per layer + logits.txt in out/apl/MODEL/, resumes after
# the last finished layer (delete the dumps to start over), and stops after
# LAST_LAYER if given. APL_BIN overrides the interpreter.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

mode=${1:?usage: apl_ref.sh dq\|run\|cmp MODEL [ARG]}
name=${2:?usage: apl_ref.sh dq\|run\|cmp MODEL [ARG]}
model="$apl_root/input/models/$name"
base=${name%.gguf}
work="$apl_out/apl/$base"
src="$apl_root/src/apl"
mkdir -p "$work"
"$apl_out/host/gguf_apl" "$model" >"$work/index.apl"
aplrun() { "$APL_BIN" --silent --noColor --noCIN -f "$1"; }

case $mode in
dq)
    # Sample rows of a Q8_0 matrix (the token embedding: large row numbers
    # included), a Q8_0 layer matrix, and two F32 tensors, in APL and in C.
    checks=("token_embd.weight 0 2" "token_embd.weight 248000 3" "blk.0.attn_qkv.weight 100 2"
            "blk.0.ssm_a 0 1" "blk.0.ssm_conv1d.weight 5 2")
    bad=0
    for c in "${checks[@]}"; do
        read -r t r0 n <<<"$c"
        script="$work/dq.apl"
        {
            cat "$work/index.apl" "$src/gguf_read.apl"
            echo "FH←⎕FIO[3] GGUF_PATH"
            if [[ $t == *ssm_* ]]; then
                echo "W←Weight '$t'"
                echo "⍕,((¯2↑1,⍴W)⍴W)[$r0+⍳$n;]"
            else
                echo "⍕,(Find '$t') Q8Rows $r0 $n"
            fi
            echo "⊣⎕FIO[4] FH"
            echo ")OFF"
        } >"$script"
        aplrun "$script" >"$work/dq-apl.txt" 2>&1
        "$apl_out/host/gguf_rows" "$model" "$t" "$r0" "$n" >"$work/dq-c.txt"
        printf '%-28s rows %s+%s: ' "$t" "$r0" "$n"
        "$apl_out/host/numcmp" "$work/dq-apl.txt" "$work/dq-c.txt" 1e-15 || bad=1
    done
    exit "$bad"
    ;;
run)
    last=${3:-¯1}
    tokens=$(/usr/bin/grep -m1 '^prompt ' "$apl_out/ref/$base/tokens.txt" | cut -d' ' -f2-)
    [[ -n $tokens ]] || { echo "apl_ref.sh: no prompt tokens: run make baseline first" >&2; exit 2; }
    {
        cat "$work/index.apl" "$src/qwen35.apl" "$src/gguf_read.apl"
        echo "TOKENS←,$tokens"
        echo "OUTDIR←'$work'"
        echo "LAST_LAYER←$last"
        cat "$src/ref_harness.apl"
    } >"$work/run.apl"
    aplrun "$work/run.apl"
    ;;
cmp)
    golden=${3:?usage: apl_ref.sh cmp MODEL GOLDEN_DIR}
    bad=0
    for f in "$golden"/*.txt; do
        printf '%-12s ' "${f##*/}"
        "$apl_out/host/numcmp" "$work/${f##*/}" "$f" 1e-12 1e-12 || bad=1
    done
    exit "$bad"
    ;;
*)
    echo "apl_ref.sh: unknown mode $mode" >&2
    exit 2
    ;;
esac
