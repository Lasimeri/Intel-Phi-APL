#!/usr/bin/env bash
# parse_oracle.sh: does each stage of the compiler's front end keep the meaning of
# qwen35.apl? Each stage's form is written back as APL and run with the original on
# a small synthetic model (tests/apl/small_model.apl) in GNU APL. GNU APL is the
# oracle: every output at ⎕PP←17 must be byte-identical.
#   rendered   the parser's trees, fully parenthesized
#   flat       the flat operation table, one operation per line
#   inlined    each entry function fully inlined, map regions as loops
# Exit 0 pass, 1 fail. Prints one summary line.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"
w="$apl_out/apl/parse-oracle"
mkdir -p "$w"
aplrun() { "$APL_BIN" --silent --noColor --noCIN -f "$1"; }
head='⎕IO←0
⎕CT←0
⎕PP←17
⎕PW←1000'
{
    printf '%s\n' "$head"
    cat "$apl_root/src/apl/compiler/tk.apl" "$apl_root/src/apl/compiler/ps.apl" \
        "$apl_root/src/apl/compiler/fl.apl" "$apl_root/src/apl/compiler/in.apl" "$apl_root/tests/apl/ps_render.apl"
    cat <<APL
SRC←19 ⎕CR ⎕UCS ⎕FIO[26] '$apl_root/src/apl/qwen35.apl'
PSProgram SRC
⎕←'parse:' (⍴FN_NAME) 'functions' (⍴ST_ROOT) 'statements' (⍴PN_K) 'nodes' (⍴PS_BAD) 'unreduced'
FH←'w' ⎕FIO[3] '$w/rendering.apl'
⊣(∊RProgram,¨TK_LF) ⎕FIO[23] FH
⊣⎕FIO[4] FH
FLProgram
⎕←'flat:' (⍴OP_V) 'ops'
FH←'w' ⎕FIO[3] '$w/flattening.apl'
⊣(∊RFlat,¨TK_LF) ⎕FIO[23] FH
⊣⎕FIO[4] FH
∇WriteEntry E;T
 INEntry E
 ⎕←'inlined:' E (⍴I_F) 'ops' (⍴IN_FN) 'instances' (⍴IN_BAD) 'unresolved'
 ⊣(∊(RInlined E),¨TK_LF) ⎕FIO[23] FH
∇
FH←'w' ⎕FIO[3] '$w/inlining.apl'
WriteEntry 'DeltaLayer'
WriteEntry 'AttnLayer'
WriteEntry 'MtpLayer'
WriteEntry 'Hidden'
WriteEntry 'DraftLogits'
WriteEntry 'Accepted'
WriteEntry 'DraftNext'
WriteEntry 'Logits'
WriteEntry 'Next'
⊣⎕FIO[4] FH
)OFF
APL
} >"$w/render.apl"
log=$(aplrun "$w/render.apl" 2>&1 || true)
summary=$(/usr/bin/grep -a -E "parse:|flat:|inlined:" <<<"$log" | tr -s " " | tr '\n' ';')
# ⍺ and ⍵ cannot be ∇ header names: the flat form renames them
sed -i -e 's/⍺/AL_/g' -e 's/⍵/OM_/g' "$w/flattening.apl"
for v in original rendered flat inlined; do
    src="$apl_root/src/apl/qwen35.apl"
    [[ $v == rendered ]] && src="$w/rendering.apl"
    [[ $v == flat ]] && src="$w/flattening.apl"
    [[ $v == inlined ]] && src="$w/inlining.apl"
    { printf '%s\n' "$head"; cat "$apl_root/tests/apl/cell.apl" "$src" "$apl_root/tests/apl/small_model.apl"; } >"$w/$v.apl"
    aplrun "$w/$v.apl" >"$w/$v.out" 2>&1
done
if cmp -s "$w/original.out" "$w/rendered.out" && cmp -s "$w/original.out" "$w/flat.out" && cmp -s "$w/original.out" "$w/inlined.out" &&
    ! /usr/bin/grep -a -q 'ERROR' "$w/original.out"; then
    echo "$summary original == rendered == flat == inlined, $(wc -w <"$w/original.out") numbers"
    exit 0
fi
echo "$summary original, rendered, flat, inlined DIFFER or errored (see $w)"
exit 1
