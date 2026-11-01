#!/usr/bin/env bash
# baseline.sh: the llama.cpp reference on the host CPU for each model in
# input/models (or the ones named as arguments): tok/s for the prompt (one
# batch) and for greedy decode, plus the token IDs and the first steps'
# logits the card engine is verified against. CPU only: the GPU is not
# touched (CUDA_VISIBLE_DEVICES is emptied). Settings are fixed so runs are
# comparable: the shared prompt, BASE_GEN generated tokens, BASE_THREADS
# threads, 1 warm-up + BASE_REPEATS repeats in one process.
#   reference data   out/ref/MODEL/{tokens.txt,logits.f32,text.txt}
#   run record       logs/baseline/DATE/HHMMSS-MODEL.txt (stamp, host load, output)
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

: "${BASE_GEN:=128}" "${BASE_THREADS:=8}" "${BASE_REPEATS:=3}"
tool="$apl_out/host/ref_llama"
[[ -x $tool ]] || { echo "baseline.sh: $tool missing (make host with a llama.cpp build at LLAMA_DIR)" >&2; exit 2; }

models=("$@")
if ((${#models[@]} == 0)); then
    while read -r _ name; do models+=("$name"); done <"$apl_root/input/models/SHA256SUMS"
fi

rec_dir="$apl_root/logs/baseline/$(date +%F)"
mkdir -p "$rec_dir"
for name in "${models[@]}"; do
    m="$apl_root/input/models/$name"
    base=${name%.gguf}
    ref="$apl_out/ref/$base"
    mkdir -p "$ref"
    rec="$rec_dir/$(date +%H%M%S)-$base.txt"
    {
        apl_stamp
        printf 'host     %s, load %s\n' "$(/usr/bin/grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')" "$(cut -d' ' -f1-3 /proc/loadavg)"
        printf 'llama    %s (git %s), CPU only\n' "$LLAMA_DIR" "$(git -C "$LLAMA_DIR" describe --always --dirty 2>/dev/null || echo unknown)"
        echo
        CUDA_VISIBLE_DEVICES= "$tool" "$m" "$ref" "$BASE_GEN" "$BASE_THREADS" "$BASE_REPEATS" "$APL_PROMPT" 2>/dev/null
        echo "reference data: ${ref#"$apl_root"/}"
    } | tee "$rec"
    echo "record: ${rec#"$apl_root"/}"
done
