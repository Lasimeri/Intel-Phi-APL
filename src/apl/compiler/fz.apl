⍝ fz.apl: fusion. A chain of elementwise rows on the host becomes one loop: each row a line of a
⍝ per-lane sequence, the same operations in the same order, so the same bits, with one store at
⍝ the end. A value all of whose readers are in the chain never reaches memory: this is what the
⍝ scalar functions of qwen35.apl are written for, one uniform node per lane instead of a pass per
⍝ node. EMPlan's EMAlloc then runs again, because a fused row reads its inputs at the root.
⍝
⍝ E_FZR[Q]: the root row Q was fused into, or ¯1. E_FZS[Q]: a root's rows, ascending; empty if Q
⍝ is no root. E_FZC[P]: the rows reading P. E_FZFN: the entry's generated functions.

∇EMFzClear;N
 N←⍴I_F
 E_FZR←N⍴¯1
 E_FZS←N⍴⊂⍳0
 E_FZD←N⍴¯1
 E_FZAUX←N⍴0
 E_FZTAP←N⍴0
∇

∇EMFzReaders;N;Q;R;K;P
 N←⍴I_F
 E_FZC←N⍴⊂⍳0
 Q←0
NEXT:→(Q≥N)/0
 R←SCReads Q
 K←0
EACH:→(K≥⍴R)/STEP
 P←SHPos R[K]
 E_FZC[P]←⊂(P⊃E_FZC),Q
 K←K+1
 →EACH
STEP:Q←Q+1
 →NEXT
∇

⍝ An elementwise row of arrays on the host: dyads, or the monads over an array
∇Z←EMFzIsEW Q;F
 Z←0
 →(S_K[Q]∨S_I[Q]∨E_FUSED[Q]∨E_DESTI[Q]∨E_KV[Q]∨E_VIEW[Q]∨E_SIDE[Q]≠1)/0
 →(0=↑⍴Q⊃S_S)/0
 F←Q⊃I_F
 Z←(I_A[Q]≥0)∧(⊂F)∊EM_EWF
 →Z/0
 Z←(I_A[Q]<0)∧(⊂F)∊EM_MONF
∇

⍝ May row R join the chain of root Q? SET LS Q BAR are the caller's. Only if every reader is
⍝ already in the chain, and no state is written or side crossed before the root reads for it.
∇Z←EMFzFits R
 Z←0
 →(R∊SET)/0
 →(E_FZR[R]≥0)/0
 →(~E_BUF[R])/0
 →(~(EMFzIsEW R)∨EMFzIsTaps R)/0
 →((R⊃S_S)≢LS)/0
 →(I_R[R]≠I_R[Q])/0
 →(~∧/(R⊃E_FZC)∊SET)/0
 →(~EMFzIsTaps R)/PLAIN
 R←SHPos I_A[SHPos I_B[R]]
PLAIN:Z←~∨/BAR[R+1+⍳0⌈Q-R+1]
∇

⍝ The chain of root Q, to a fixed point: a value read twice joins once both readers have
∇Z←LS EMFzGrow Q;SET;I;CH;O;K;R
 SET←,Q
AGAIN:CH←0
 I←0
MEMBER:→(I≥⍴SET)/PASS
 O←I_A[SET[I]],I_B[SET[I]]
 O←(O≥0)/O
 K←0
OPER:→(K≥⍴O)/MNEXT
 R←SHPos O[K]
 →(~EMFzFits R)/ONEXT
 SET←SET,R
 SET←SET,EMFzAux R
 CH←1
ONEXT:K←K+1
 →OPER
MNEXT:I←I+1
 →MEMBER
PASS:→CH/AGAIN
 Z←SET[⍋SET]
∇

⍝ From the last row up, so a root takes the longest chain
∇EMFzPlan;N;Q;BAR;SET;LS;F
 EMFzClear
 EMFzReaders
 N←⍴I_F
 BAR←(I_F∊'set' 'setix')∨E_SIDE=0
 Q←N-1
ROOT:→(Q<0)/DONE
 →(E_FZR[Q]≥0)/NEXT
 F←Q⊃I_F
 →(EMFzIsEW Q)/EW
 →(F≡'(+/)')/SUM
 →NEXT
EW:LS←Q⊃S_S
 SET←LS EMFzGrow Q
 →KEEP
SUM:→((E_SIDE[Q]≠1)∨S_K[Q]∨S_I[Q])/NEXT
 LS←(SHPos I_B[Q])⊃S_S
 →((1≠⍴LS)∨(¯1∊LS)∨0≠8|×/LS)/NEXT
 SET←LS EMFzGrow Q
 →(1≥⍴SET)/NEXT
KEEP:E_FZS[Q]←⊂SET
 E_FZR[(SET≠Q)/SET]←Q
NEXT:Q←Q-1
 →ROOT
DONE:EMFzInPlace
 E_BUF[⍸(E_FZR≥0)∨E_FZD≥0]←0
∇

⍝ A min with a constant under exp: exp8 clamps its exponent at 127.99 × ln 2 already, so the row
⍝ exists only for GNU APL's F64, and is an alias here: the same bits
∇Z←EMFzDeadMin S;P
 Z←0
 →((S⊃I_F)≢,'⌊')/0
 →(I_A[S]<0)/0
 P←SHPos I_A[S]
 →((~S_K[P])∨0≠↑⍴P⊃S_S)/0
 →(128>↑P⊃S_V)/0
 Z←∧/{((⍵⊃I_F)≡,'*')∧I_A[⍵]<0}¨S⊃E_FZC
∇

⍝ The per-lane name of value V within the chain: SET and L are the caller's
∇Z←EMFzName V;P
 P←SHPos V
 →(P∊SET)/ROW
 Z←'v',⍕L⍳P
 →0
ROW:Z←'r',⍕P
∇

⍝ One row as a line of the sequence
∇Z←EMFzLine S;F;B;A;E
 →(~E_FZTAP[S])/NODE
 E←EMFzTapLine S
 →DONE
NODE:F←S⊃I_F
 B←EMFzName I_B[S]
 →(I_A[S]≥0)/DYAD
 →(F≡,'-')/NEG
 →(F≡,'÷')/RCP
 →(F≡,'*')/EXP
 E←'log8(',B,')'
 →DONE
EXP:E←'exp8(',B,')'
 →DONE
RCP:E←'_mm256_div_ps(_mm256_set1_ps(1.0f), ',B,')'
 →DONE
NEG:E←'_mm256_sub_ps(_mm256_setzero_ps(), ',B,')'
 →DONE
DYAD:→(EMFzDeadMin S)/ALIAS
 A←EMFzName I_A[S]
 E←'_mm256_',((EM_EWF⍳⊂F)⊃EM_EWL),'_ps(',A,', ',B,')'
 →DONE
ALIAS:E←B
DONE:Z←'  const __m256 r',(⍕S),' = ',E,';'
∇

⍝ The chain of root Q: its function joins E_FZFN, Z is the call
∇Z←EMFzEmit Q;SET;ID;SUM;LS;L;LV;I;O;K;P;FIELDS;INIT;TOP;LOAD;BODY;T;COUNT;NAME;A0;SC;V
 SET←Q⊃E_FZS
 V←⍕I_V[Q]
 ID←'FZ',EM_E,'_',V
 SUM←(Q⊃I_F)≡'(+/)'
 LS←SUM⊃(Q⊃S_S)((SHPos I_B[Q])⊃S_S)
 COUNT←EMCount LS
 ⍝ the leaves: what the chain reads from outside, each once
 L←LV←⍳0
 I←0
LEAFROW:→(I≥⍴SET)/LEAVES
 O←I_A[SET[I]],I_B[SET[I]]
 O←(O≥0)/O
 K←0
LEAFOP:→(K≥⍴O)/LEAFNEXT
 P←SHPos O[K]
 →((P∊SET)∨P∊L)/LEAFSKIP
 L←L,P
 LV←LV,O[K]
LEAFSKIP:K←K+1
 →LEAFOP
LEAFNEXT:I←I+1
 →LEAFROW
LEAVES:FIELDS←INIT←''
 TOP←LOAD←0⍴⊂''
 K←0
FIELD:→(K≥⍴L)/LINES
 T←⍕K
 INIT←INIT,', ',EMOp LV[K]
 SC←0=↑⍴L[K]⊃S_S
 →SC/SCALAR
 FIELDS←FIELDS,' const float *x',T,';'
 TOP←TOP,⊂' const float *const x',T,' = a->x',T,';'
 LOAD←LOAD,⊂'  UNUSED const __m256 v',T,' = _mm256_load_ps(x',T,' + j);'
 →FNEXT
SCALAR:FIELDS←FIELDS,' float x',T,';'
 TOP←TOP,⊂' UNUSED const __m256 v',T,' = _mm256_set1_ps(a->x',T,');'
FNEXT:K←K+1
 →FIELD
LINES:K←0
TAPS:→(K≥⍴SET)/TAPSDONE
 →(~E_FZTAP[SET[K]])/TAPSNEXT
 TOP←TOP,EMFzTapTop SET[K]
 FIELDS←FIELDS,' const long *l',(⍕SET[K]),';'
 INIT←INIT,', ',EMFzTapList SET[K]
TAPSNEXT:K←K+1
 →TAPS
TAPSDONE:BODY←EMFzLine¨((~E_FZAUX[SET])∧(~SUM)∨SET≠Q)/SET
 →SUM/SUMFN
 ⍝ whole 16-lane blocks, as rt_ew
 NAME←' struct ',ID,'_a a',V
 E_FZFN←E_FZFN,⊂'struct ',ID,'_a { long n; float *z;',FIELDS,' };'
 E_FZFN←E_FZFN,(⊂'static void ',ID,'(void *arg, UNUSED int tid, UNUSED int nt)'),⊂,'{'
 E_FZFN←E_FZFN,⊂' const struct ',ID,'_a *a = arg;'
 E_FZFN←E_FZFN,⊂' const long blocks = (a->n + 15) / 16, i0 = blocks * tid / nt * 16, i1 = blocks * (tid + 1) / nt * 16;'
 E_FZFN←E_FZFN,⊂' float *const z = a->z;'
 E_FZFN←E_FZFN,TOP,⊂' for (long j = i0; j < i1; j += 8) {'
 E_FZFN←E_FZFN,LOAD,BODY,(⊂'  _mm256_store_ps(z + j, r',(⍕Q),');'),(⊂' }'),⊂,'}'
 →(E_FZD[Q]<0)/OWNDEST
 Z←NAME,' = {',COUNT,', V',(⍕I_V[E_FZD[Q]]),' + IX',(⍕I_I[E_FZD[Q]]),' * ',COUNT,INIT,'};',TK_LF
 →CALL
OWNDEST:Z←NAME,' = {',COUNT,', V',V,INIT,'};',TK_LF
CALL:
 →((¯1∊LS)∨EM_PAR_LANES≤×/LS)/TEAM
 Z←Z,' ',ID,'(&a',V,', 0, 1);'
 →0
TEAM:Z←Z,' rt_fz(R, ',ID,', &a',V,', ',COUNT,');'
 →0
SUMFN:E_FZFN←E_FZFN,⊂'struct ',ID,'_a { long n;',FIELDS,' };'
 E_FZFN←E_FZFN,(⊂'static inline float ',ID,'(const struct ',ID,'_a *a)'),⊂,'{'
 E_FZFN←E_FZFN,TOP,⊂' __m256 acc = _mm256_setzero_ps();'
 E_FZFN←E_FZFN,⊂' for (long j = 0; j < a->n; j += 8) {'
 E_FZFN←E_FZFN,LOAD,BODY,(⊂'  acc = _mm256_add_ps(acc, ',(EMFzName I_B[Q]),');'),(⊂' }'),(⊂' return hsum8(acc);'),⊂,'}'
 Z←' struct ',ID,'_a a',V,' = {',COUNT,INIT,'};',TK_LF,' UNUSED const float s',V,' = ',ID,'(&a',V,');'
∇

⍝ ---- a depthwise conv: +⌿ of (a list of a state's rows)×(a matrix) is one node as well, its
⍝ array of copies and its array of products never built. Same products, same order of sums as
⍝ rt_redfirst, so the same bits.
∇Z←EMFzIsTaps Q;P;I
 Z←0
 →((Q⊃I_F)≢'(+⌿)')/0
 →(S_K[Q]∨E_SIDE[Q]≠1)/0
 P←SHPos I_B[Q]
 →(((P⊃I_F)≢,'×')∨(I_A[P]<0)∨(~E_BUF[P])∨(E_SIDE[P]≠1)∨(1≠⍴P⊃E_FZC)∨2≠⍴P⊃S_S)/0
 →(¯1∊P⊃S_S)/0
 I←SHPos I_A[P]
 →(((I⊃I_F)≢'index')∨(~E_BUF[I])∨(E_SIDE[I]≠1)∨(1≠⍴I⊃E_FZC)∨2≠⍴I⊃S_S)/0
 →(2≠⍴I⊃I_X)/0
 Z←S_K[SHPos I_B[P]]≤2≠↑⍴(SHPos I_B[P])⊃S_S
 →(~Z)/0
 Z←(~S_K[SHPos I_A[I]])∧2=↑⍴(SHPos I_B[P])⊃S_S
∇

⍝ The rows a conv node takes along, which write no line
∇Z←EMFzAux R;P
 Z←⍳0
 →(~EMFzIsTaps R)/0
 P←SHPos I_B[R]
 Z←P,SHPos I_A[P]
 E_FZAUX[Z]←1
 E_FZTAP[R]←1
∇

∇Z←EMFzTapList S;P;I
 P←SHPos I_B[S]
 I←SHPos I_A[P]
 Z←↑(((SHPos I_A[I])⊃S_S)[0]) EMList (I⊃I_X)[0]
∇

⍝ One row pointer per tap, before the loop
∇Z←EMFzTapTop S;P;I;AK;C;R;T
 P←SHPos I_B[S]
 I←SHPos I_A[P]
 AK←⍕L⍳SHPos I_A[I]
 (R C)←P⊃S_S
 T←⍕S
 Z←,⊂' const long *const l',T,' = a->l',T,';'
 Z←Z,{' const float *const t',T,'_',(⍕⍵),' = x',AK,' + l',T,'[',(⍕⍵),'] * ',(⍕C),';'}¨⍳R
∇

∇Z←EMFzTapLine S;P;W;C;R;T;K;Y
 P←SHPos I_B[S]
 W←⍕L⍳SHPos I_B[P]
 (R C)←P⊃S_S
 T←⍕S
 K←R-1
 Z←'_mm256_mul_ps(_mm256_load_ps(t',T,'_',(⍕K),' + j), _mm256_load_ps(x',W,' + ',(⍕K×C),' + j))'
NEXT:K←K-1
 →(K<0)/0
 Y←'_mm256_mul_ps(_mm256_load_ps(t',T,'_',(⍕K),' + j), _mm256_load_ps(x',W,' + ',(⍕K×C),' + j))'
 Z←'_mm256_add_ps(',Y,', ',Z,')'
 →NEXT
∇

⍝ ---- a map's cell as the last thing a body makes: written straight into the map's row, instead
⍝ of a place of its own and a copy. Whole 16-lane blocks only, the loop's granularity.
∇EMFzInPlace;P;K;C;A
 P←⍸I_F∊⊂'}map'
 K←0
NEXT:→(K≥⍴P)/0
 C←P[K]
 A←SHPos I_A[C]
 →(0=↑⍴A⊃E_FZS)/SKIP
 →((A⊃I_F)≡'(+/)')/SKIP
 →((~E_BUF[A])∨(E_SIDE[C]≠1)∨(I_R[A]≠I_I[C])∨(¯1∊A⊃S_S)∨(0≠16|×/A⊃S_S)∨(,C)≢A⊃E_FZC)/SKIP
 E_FZD[A]←C
SKIP:K←K+1
 →NEXT
∇
