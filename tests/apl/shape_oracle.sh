#!/usr/bin/env bash
# shape_oracle.sh: is the compiler's shape pass (src/apl/compiler/sh.apl) right? The
# inlined program of each entry runs on the small synthetic model with a print of every
# value's real shape; each must equal what the compiler inferred, T being exactly p+1,
# and every row must have run. Exit 0 pass, 1 fail. Prints one summary line.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"
w="$apl_out/apl/shape-oracle"
mkdir -p "$w"
aplrun() { "$APL_BIN" --silent --noColor --noCIN -f "$1"; }
head='⎕IO←0
⎕CT←0
⎕PP←17
⎕PW←1000'
{
    printf '%s\n' "$head"
    cat "$apl_root/src/apl/compiler/tk.apl" "$apl_root/src/apl/compiler/ps.apl" \
        "$apl_root/src/apl/compiler/fl.apl" "$apl_root/src/apl/compiler/in.apl" \
        "$apl_root/src/apl/compiler/sh.apl" "$apl_root/tests/apl/ps_render.apl"
    sed -n '/^D←32$/,/^X←Rnd 3,D$/p' "$apl_root/tests/apl/small_model.apl"
    cat <<APL
SRC←19 ⎕CR ⎕UCS ⎕FIO[26] '$apl_root/src/apl/qwen35.apl'
PSProgram SRC
FLProgram
SH_W←'W_EH' 'W_ENORM' 'W_HNORM' 'W_SHN' 'W_ANORM' 'W_PNORM' 'W_UP' 'W_GATE2' 'W_DOWN' 'W_QKV' 'W_GATE' 'W_ALPHA' 'W_BETA' 'W_DT' 'W_A' 'W_SNORM' 'W_CONV' 'W_OUT' 'W_Q' 'W_K' 'W_V' 'W_O' 'W_QN' 'W_KN' 'W_ONORM' 'W_OUTPUT'
SH_WS←{⍴⍎⍵}¨SH_W
SH_ST←,¨'WIN' 'S' 'KC' 'VC'
SH_RI←,¨'p' 'TOK' 'DR' 'AG'
SH_STS←{⍴⍎⍵}¨SH_ST
SH_C←,¨'D' 'FF' 'HK' 'HV' 'ST' 'HQ' 'HKV' 'HD' 'NK' 'NV' 'CTX' 'RMS_EPS' 'ROPE_N' 'ROPE_COS' 'ROPE_SIN' 'KV_OF' 'NB' 'WL' 'SL' 'HOT'
∇Z←E SHArgs Y
 Z←((⊂,'p'),⊂,'X')((⊂⍳0),⊂NB,D)
 →((⊂E)∊'DeltaLayer' 'AttnLayer')/0
 Z←((⊂,'p'),⊂,'X')((⊂⍳0),⊂NB,2×D)
 →(E≡'MtpLayer')/0
 Z←(,⊂,'X')(,⊂NB,D)
 →(~(⊂E)∊'Next' 'DraftNext' 'Accepted')/0
 Z←((⊂'DR'),⊂'AG')((⊂,3),⊂,3)
 →(E≡'Accepted')/0
 Z←(,⊂,'Y')(,⊂,HOT)
 →(E≡'DraftNext')/0
 Z←(,⊂,'⍵')(,⊂,64)
∇
∇WriteEntry E
 INEntry E
 SH_RT←E SHArgs 0
 SHEntry
 ⎕←'entry:' E (⍴I_F) 'ops' (+/S_K) 'known' (+/S_I) 'runtime ints' (⍴SH_BAD) 'no rule' (∪SH_BAD)
 ⊣(∊(RInlinedS E),¨TK_LF) ⎕FIO[23] FH
 ⊣(∊(E SHLines 0),¨TK_LF) ⎕FIO[23] FG
∇
FH←'w' ⎕FIO[3] '$w/program.apl'
FG←'w' ⎕FIO[3] '$w/inferred.txt'
WriteEntry 'DeltaLayer'
WriteEntry 'AttnLayer'
WriteEntry 'MtpLayer'
WriteEntry 'Hidden'
WriteEntry 'DraftLogits'
WriteEntry 'DraftNext'
WriteEntry 'Accepted'
WriteEntry 'Logits'
WriteEntry 'Next'
⊣⎕FIO[4] FH
⊣⎕FIO[4] FG
)OFF
APL
} >"$w/infer.apl"
log=$(aplrun "$w/infer.apl" 2>&1 || true)
printf '%s\n' "$log" >"$w/infer.log"
/usr/bin/grep -a -q 'ERROR' <<<"$log" && { echo "shape pass errored (see $w/infer.log)"; exit 1; }
{ printf '%s\n' "$head"; cat "$apl_root/tests/apl/cell.apl" "$w/program.apl" "$apl_root/tests/apl/small_model.apl"; } >"$w/run.apl"
aplrun "$w/run.apl" >"$w/run.out" 2>&1
/usr/bin/grep -a -q 'ERROR' "$w/run.out" && { echo "program errored (see $w/run.out)"; exit 1; }
unruled=$(awk '/entry:/ && !/ 0  no rule/ { n++ } END { print n + 0 }' <<<"$log")
result=$(awk -f "$apl_root/tests/apl/shape_check.awk" "$w/inferred.txt" "$w/run.out") && ok=1 || ok=0
echo "$result" >"$w/check.txt"
echo "$result" | tail -1
[[ $ok == 1 && $unruled == 0 ]]
