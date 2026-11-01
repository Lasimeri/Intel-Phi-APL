#!/usr/bin/env bash
# build.sh: build the APL engine. Arguments: any of `host`, `card` (both
# when none). Everything goes to out/; nothing outside the project is
# written. Every compile runs with -Wall -Wextra -Werror; every card binary
# must pass phi-isa-audit with 0 illegal and 0 suspect instructions.
#
#   host  out/host/gguf_info              GGUF reader + completeness check
#         out/host/gguf_apl               the tensor index as a GNU APL script
#         out/host/gguf_rows, numcmp      test tools: C dequant, number compare
#         out/host/t_knc_half             host test of knc.h's half decoder
#         out/host/fma-gate, fma-ref      gate A: host model, independent reference
#         out/host/upconv, upconv_cmp     gate 2.1: host model, judge
#   card  out/card/*.o                    VPU kernels (KNC clang, -march=knc)
#         out/card/fma-gate, upconv       card programs (knc-cc, static)
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

src="$apl_root/src"
targets=("$@")
((${#targets[@]})) || targets=(host card)

build_host() {
    local o="$apl_out/host"
    mkdir -p "$o"
    host_cc -o "$o/gguf_info" "$src/host/gguf_info.c" "$src/host/gguf_reader.c"
    host_cc -I "$src/card" -o "$o/apl_feed" "$src/host/apl_feed.c"
    host_cc -I "$src/card" -o "$o/cross_host" "$src/bench/cross_host.c"
    host_cc -o "$o/gguf_apl" "$src/host/gguf_apl.c" "$src/host/gguf_reader.c"
    host_cc -o "$o/gguf_rows" "$apl_root/tests/host/gguf_rows.c" "$src/host/gguf_reader.c"
    host_cc -I "$apl_root/tests/host" -o "$o/numcmp" "$apl_root/tests/host/numcmp.c" -lm
    host_cc -I "$apl_root/tests/host" -o "$o/txt2f32" "$apl_root/tests/host/txt2f32.c"
    host_cc -o "$o/logit_cmp" "$apl_root/tests/host/logit_cmp.c" -lm
    host_cc -o "$o/t_knc_half" "$apl_root/tests/host/t_knc_half.c"
    host_cc -o "$o/fma-gate" "$src/gate/fma_main.c" "$src/gate/fma_kernel.c" -lm
    host_cc -o "$o/fma-ref" "$src/gate/fma_ref.c" -lm
    host_cc -o "$o/upconv" "$src/gate/upconv_main.c" "$src/gate/upconv_kernel.c" -lm
    host_cc -o "$o/upconv_cmp" "$src/gate/upconv_cmp.c" -lm
    host_cc -pthread -o "$o/gemv_bench" "$src/bench/gemv_bench.c" "$src/card/gemv.c" "$src/card/gemm.c" "$src/card/team.c" -lm
    host_cc -pthread -I "$src/bench" -o "$o/real_gemv" "$src/bench/real_gemv.c" "$src/card/gemv.c" "$src/card/team.c" "$src/card/stream.c" -lm
    echo "host: gguf_info gguf_apl gguf_rows numcmp logit_cmp t_knc_half fma-gate fma-ref upconv upconv_cmp gemv_bench real_gemv apl_feed txt2f32 -> out/host"
    # The offline weight analysis, on the GPU: built when OpenCL's header is there
    if [[ -r $apl_cl_include/CL/cl.h ]]; then
        host_cc -I "$apl_cl_include" -o "$o/spectrum" "$src/gpu/spectrum.c" -lOpenCL -llapack -lm
    fi
    # The llama.cpp reference (test only): built when an existing llama.cpp
    # build is there; linked against it read-only.
    if [[ -r "$LLAMA_DIR/include/llama.h" && -e "$LLAMA_DIR/build/bin/libllama.so" ]]; then
        host_cc -I "$LLAMA_DIR/include" -I "$LLAMA_DIR/ggml/include" -o "$o/ref_llama" "$src/host/ref_llama.c" \
            -L "$LLAMA_DIR/build/bin" -Wl,-rpath,"$LLAMA_DIR/build/bin" -lllama
        echo "host: ref_llama (against $LLAMA_DIR) -> out/host"
    else
        echo "host: ref_llama skipped (no llama.cpp build at LLAMA_DIR=$LLAMA_DIR)"
    fi
}

build_card() {
    local o="$apl_out/card"
    mkdir -p "$o"
    vpu_cc -c -o "$o/fma_kernel.o" "$src/gate/fma_kernel.c"
    vpu_cc -c -o "$o/upconv_kernel.o" "$src/gate/upconv_kernel.c"
    card_cc -o "$o/fma-gate" "$src/gate/fma_main.c" "$o/fma_kernel.o"
    card_cc -o "$o/upconv" "$src/gate/upconv_main.c" "$o/upconv_kernel.o"
    vpu_cc -c -o "$o/team.o" "$src/card/team.c"
    vpu_cc -c -o "$o/gemv.o" "$src/card/gemv.c"
    vpu_cc -c -o "$o/gemm.o" "$src/card/gemm.c"
    vpu_cc -c -o "$o/stream.o" "$src/card/stream.c"
    card_cc -I "$src/bench" -o "$o/gemv_bench" "$src/bench/gemv_bench.c" "$o/gemv.o" "$o/gemm.o" "$o/team.o"
    card_cc -I "$src/bench" -o "$o/mem_order" "$src/bench/mem_order.c" "$o/team.o"
    card_cc -I "$src/bench" -o "$o/real_gemv" "$src/bench/real_gemv.c" "$o/gemv.o" "$o/team.o" "$o/stream.o"
    vpu_cc -c -o "$o/cdma.o" "$src/card/cdma.c"
    vpu_cc -c -o "$o/link.o" "$src/card/link.c"
    card_cc -pthread -I "$src/bench" -o "$o/cross_card" "$src/bench/cross_card.c" "$o/cdma.o" "$o/link.o"
    audit_clean "$o/fma-gate"
    audit_clean "$o/upconv"
    audit_clean "$o/gemv_bench"
    audit_clean "$o/mem_order"
    audit_clean "$o/real_gemv"
    audit_clean "$o/cross_card"
    echo "card: fma-gate upconv gemv_bench real_gemv -> out/card"
}

for t in "${targets[@]}"; do
    case $t in
    host) build_host ;;
    card) build_card ;;
    *) echo "build.sh: unknown target '$t' (host, card)" >&2; exit 2 ;;
    esac
done
