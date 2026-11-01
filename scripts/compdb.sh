#!/usr/bin/env bash
# compdb.sh: write compile_commands.json for clangd and prove it clean.
# Every C file gets the host flags (clang spelling) plus src/card and
# src/gate on the include path: knc.h then shows clangd its host model,
# which is the same source the card build compiles under __KNC__. The card
# build itself is checked by the KNC clang with -Werror in build.sh.
#
# Verification: clangd --check on every .c and .h file. A diagnostic is a
# line "E[time] [code] Line N: ..." (or W[...]); clangd also prints
# "E[time]     tweak: X ==> FAIL" lines and counts them in its "All checks
# completed, N errors" total, but those are its own refactoring self-tests
# on the file, not diagnostics of the code, so only the [code] lines count.
# Exit 0 only when every file has none.
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

db="$apl_root/compile_commands.json"
mapfile -t sources < <(cd "$apl_root" && find src tests -name '*.c' | LC_ALL=C sort)
mapfile -t checked < <(cd "$apl_root" && find src tests -name '*.[ch]' | LC_ALL=C sort)

{
    echo "["
    for i in "${!sources[@]}"; do
        f=${sources[$i]}
        sep=","
        ((i == ${#sources[@]} - 1)) && sep=""
        printf '  {"directory": "%s", "file": "%s", "arguments": ["clang", "-std=gnu11", "-O2", "-Wall", "-Wextra", "-I'"$apl_cl_include"'", "-Werror", "-ffp-contract=off", "-mavx2", "-mfma", "-I", "src/card", "-I", "src/gate", "-I", "src/host", "-I", "src/bench", "-I", "tests/host", "-I", "%s/include", "-I", "%s/ggml/include", "-c", "%s"]}%s\n' \
            "$apl_root" "$f" "$LLAMA_DIR" "$LLAMA_DIR" "$f" "$sep"
    done
    echo "]"
} >"$db"
echo "compile_commands.json: ${#sources[@]} entries; clangd --check over ${#checked[@]} files"

bad=0
for f in "${checked[@]}"; do
    log=$(cd "$apl_root" && clangd --check="$f" 2>&1) || true
    if ! /usr/bin/grep -a -q 'All checks completed' <<<"$log"; then
        printf 'clangd did not finish  %s\n' "$f"
        bad=1
        continue
    fi
    diags=$(/usr/bin/grep -a -E '^[EW]\[[0-9:.]+\] \[' <<<"$log" || true)
    if [[ -z $diags ]]; then
        printf 'clangd 0 diagnostics  %s\n' "$f"
    else
        printf 'clangd %d diagnostics  %s\n' "$(wc -l <<<"$diags")" "$f"
        head -5 <<<"$diags"
        bad=1
    fi
done
exit "$bad"
