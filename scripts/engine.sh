#!/usr/bin/env bash
# engine.sh: the APL compiler, the engine's builds, and its runs, on one model.
#
#   engine.sh compile MODEL     qwen35.apl + the model's index -> out/gen/MODEL/*.c
#   engine.sh build MODEL       the card's halves, the host's worker, the host test model
#   engine.sh split MODEL N_GEN TPC CORES HOST_THREADS REPEATS [DUMP_DIR]
#                               on the card with the host's worker: the production form
#   engine.sh card MODEL N_GEN TPC CORES REPEATS [DUMP]
#                               on the card alone, for comparison; a DUMP lands in
#                               out/gen/MODEL/dump-card
#       PROFILE=1 with split or card runs the profile build: where a token's time goes
#   engine.sh host MODEL N_GEN THREADS REPEATS [DUMP_DIR]
#                               the host test model (TEST ONLY)
#
# MODEL: a file name in input/models without .gguf. The prompt comes from the llama.cpp
# reference (out/ref/MODEL/tokens.txt, made by make baseline). Card programs run through
# the control socket, not ssh: phictl's forward runs out of connection slots.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage="usage: engine.sh compile|build|split|card|host MODEL ..."
mode=${1:?$usage}
name=${2:?$usage}
model="$apl_root/input/models/$name.gguf"
gen="$apl_out/gen/$name"
src="$apl_root/src"
card="$apl_out/card"
tokens="$apl_out/ref/$name/tokens.txt"
card_dir=/tmp/apl
mkdir -p "$gen" "$card"

compile() {
    local out
    "$apl_out/host/gguf_apl" "$model" >"$gen/index.apl"
    {
        printf '⎕IO←0\n⎕CT←0\n⎕PP←17\n⎕PW←1000\n'
        cat "$gen/index.apl"
        echo "SRC_PATH←'$src/apl/qwen35.apl'"
        echo "OUT_PATH←'$gen'"
        cat "$src/apl/compiler/"{tk,ps,fl,in,sh,sc,em,fz,main}.apl
    } >"$gen/compile.apl"
    out=$("$APL_BIN" --silent --noColor --noCIN -f "$gen/compile.apl" 2>&1 || true)
    printf '%s\n' "$out" >"$gen/compile.log"
    if /usr/bin/grep -a -q 'ERROR' <<<"$out"; then
        printf '%s\n' "$out" | head -24
        echo "engine.sh: the compile failed (see $gen/compile.log)" >&2
        exit 1
    fi
    /usr/bin/grep -a -E 'compiled:|placement:' <<<"$out"
}

build() {
    local f objs
    for f in team gemv gemm stream cdma link rt; do
        vpu_cc -c -o "$card/$f.o" "$src/card/$f.c"
    done
    objs=("$card/rt.o" "$card/gemv.o" "$card/gemm.o" "$card/team.o" "$card/stream.o" "$card/cdma.o" "$card/link.o")
    vpu_cc -c -o "$gen/model.o" "$gen/model.c"
    vpu_cc -DRT_PROFILE -c -o "$gen/model-prof.o" "$gen/model.c"
    vpu_cc -c -o "$gen/model_card.o" "$gen/model_card.c"
    vpu_cc -DRT_PROFILE -c -o "$gen/model_card-prof.o" "$gen/model_card.c"
    card_cc -I "$src/bench" -o "$card/engine-$name" "$src/card/engine.c" "$gen/model.o" "${objs[@]}"
    card_cc -DRT_PROFILE -I "$src/bench" -o "$card/engine-$name-prof" "$src/card/engine.c" "$gen/model-prof.o" "${objs[@]}"
    card_cc -I "$src/bench" -o "$card/engine-$name-split" "$src/card/engine.c" "$gen/model_card.o" "${objs[@]}"
    card_cc -DRT_PROFILE -I "$src/bench" -o "$card/engine-$name-split-prof" "$src/card/engine.c" "$gen/model_card-prof.o" "${objs[@]}"
    audit_clean "$card/engine-$name"
    audit_clean "$card/engine-$name-prof"
    audit_clean "$card/engine-$name-split"
    audit_clean "$card/engine-$name-split-prof"
    host_cc -pthread -I "$src/bench" -o "$apl_out/host/engine-$name" "$src/card/engine.c" "$gen/model.c" \
        "$src/card/"{rt,gemv,gemm,team,stream,link,cdma}.c -lm
    host_cc -pthread -mavx2 -mfma -o "$apl_out/host/engine_host-$name" "$src/host/engine_host.c" "$gen/model_card_host.c" \
        "$src/host/rt_host.c" "$src/host/link_host.c" "$src/card/team.c" -lm
}

# The host's worker and the card in lockstep over the link; the feed first.
run_split() {
    local n_gen=${1:?N_GEN} tpc=${2:?TPC} cores=${3:?CORES} threads=${4:?HOST_THREADS} repeats=${5:?REPEATS} dump=${6:-}
    local feed host rc=0
    card_pool "$model"
    phic put "$card/engine-$name-split${PROFILE:+-prof}" "$card_dir/engine" >/dev/null
    phic put "$tokens" "$card_dir/tokens.txt" >/dev/null
    "$apl_out/host/apl_feed" "$model" >"$gen/feed.log" 2>&1 &
    feed=$!
    "$apl_out/host/engine_host-$name" "$model" "$tokens" "$n_gen" "$threads" "$repeats" ${dump:+"$dump"} &
    host=$!
    # APL_CARD_ENV: settings for a sweep, for example "APL_PF_L1=4096 APL_NO_TILES=1"
    # shellcheck disable=SC2086
    phic run env ${APL_CARD_ENV:-APL=1} "$card_dir/engine" stream "$card_dir/tokens.txt" "$n_gen" "$tpc" "$cores" 0 >"$gen/card.log" 2>&1 || rc=$?
    wait "$feed" || rc=$?
    wait "$host" || rc=$?
    cat "$gen/feed.log" "$gen/card.log"
    return "$rc"
}

run_card() {
    local n_gen=${1:?N_GEN} tpc=${2:?TPC} cores=${3:?CORES} repeats=${4:?REPEATS} dump=${5:-}
    local feed rc=0 args=()
    card_pool "$model"
    phic put "$card/engine-$name${PROFILE:+-prof}" "$card_dir/engine" >/dev/null
    phic put "$tokens" "$card_dir/tokens.txt" >/dev/null
    "$apl_out/host/apl_feed" "$model" >"$gen/feed.log" 2>&1 &
    feed=$!
    [[ -n $dump ]] && args=("$card_dir/dump")
    phic run "$card_dir/engine" stream "$card_dir/tokens.txt" "$n_gen" "$tpc" "$cores" "$repeats" "${args[@]}" || rc=$?
    wait "$feed" || rc=$?
    cat "$gen/feed.log"
    if [[ -n $dump ]]; then
        rm -rf "$gen/dump-card"
        mkdir -p "$gen/dump-card"
        phic run tar -C "$card_dir/dump" -cf "$card_dir/dump.tar" . >/dev/null
        phic get "$card_dir/dump.tar" "$gen/dump-card.tar" >/dev/null
        tar -C "$gen/dump-card" -xf "$gen/dump-card.tar"
    fi
    return "$rc"
}

shift 2
case $mode in
compile) compile ;;
build) build ;;
split) run_split "$@" ;;
card) run_card "$@" ;;
host)
    exec "$apl_out/host/engine-$name" "$model" "$tokens" "${1:?N_GEN}" 1 "${2:?THREADS}" "${3:?REPEATS}" ${4:+"$4"}
    ;;
*)
    echo "$usage" >&2
    exit 2
    ;;
esac
