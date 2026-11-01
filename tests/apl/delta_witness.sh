#!/usr/bin/env bash
# delta_witness.sh: does qwen35.apl's DeltaHead still mean the delta rule as first written?
#   S_new <- (E x S_old) + N o.x K,  Z <- S_new +.x Q
# DeltaHead takes Z from S_old, without reading S_new. Both run in GNU APL (F64) on one state of
# generic values: no zeros, no small integers, k*A mod 1 with A the Glaisher-Kinkelin constant,
# the witness idea of arXiv:2603.21852 Sect. 2. A sieve at one point, not a proof. A wrong form
# must differ, or the test has no teeth. Exit 0 pass, 1 fail. Prints one summary line.
. "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"
w="$apl_out/apl/delta-witness"
mkdir -p "$w"
{
    printf '⎕IO←0\n⎕CT←0\n⎕PP←17\n⎕PW←1000\n'
    cat "$apl_root/src/apl/qwen35.apl"
    cat <<'APL'
Wit←{⍵⍴¯1+2×1|(⍺+1+⍳×/⍵)*1.2824271291006226}
HV←3
HK←3
ST←16
SO←1
SN←2
S←0 Wit 6,HV,ST,ST
V←100 Wit HV,ST
K←200 Wit HV,ST
Q←300 Wit HV,ST
E←400 Wit ,HV
B←500 Wit ,HV
S0←S
∇Z←OldHead h;N
 N←B[h]×V[h;]-E[h]×S[SO;h;;]+.×K[h;]
 S[SN;h;;]←(E[h]×S[SO;h;;])+N∘.×K[h;]
 Z←S[SN;h;;]+.×Q[h;]
∇
∇Z←WrongHead h;A;C;N
 A←S[SO;h;;]+.×K[h;]
 C←S[SO;h;;]+.×Q[h;]
 N←B[h]×V[h;]-E[h]×A
 Z←C+N×+/K[h;]×Q[h;]
∇
ZO←OldHead⍤0⊢⍳HV
SO1←S
S←S0
ZN←DeltaHead⍤0⊢⍳HV
SN1←S
S←S0
ZW←WrongHead⍤0⊢⍳HV
⎕←'witness' (⌈/|,ZO-ZN) (⌈/|,SO1-SN1) (⌈/|,ZO-ZW) (⌈/|,ZO)
)OFF
APL
} >"$w/run.apl"
"$APL_BIN" --silent --noColor --noCIN -f "$w/run.apl" >"$w/run.out" 2>&1
/usr/bin/grep -a -q 'ERROR' "$w/run.out" && { echo "delta witness errored (see $w/run.out)"; exit 1; }
sed 's/¯/-/g' "$w/run.out" | awk '/witness/ { z = $2 + 0; s = $3 + 0; wrong = $4 + 0; size = $5 + 0; seen = 1 }
     END { if (!seen) { print "no result"; exit 1 }
           printf "DeltaHead == first form: Z max diff %.3e, state %.3e (limit 1e-12); wrong form differs by %.3e, Z up to %.3f\n", z, s, wrong, size
           exit !(z < 1e-12 && s < 1e-12 && wrong > 1e-3) }'
