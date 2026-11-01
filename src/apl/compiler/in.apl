⍝ in.apl: inline: from the flat table of all functions (fl.apl) to one linear table
⍝ for one entry function. Rounds until no calls remain; each round replaces every
⍝ call at once by a copy of the callee's rows, its value ids offset by a new instance
⍝ number, between a `set` per argument and a read of its Z. A function applied under
⍝ the rank operator becomes a map region: the rows of its body between `map{` and
⍝ `}map`, its arguments arriving through `cell` rows: a loop over the leading axis, the
⍝ natural unit to split across threads. It is not unrolled.
⍝
⍝ Names are resolved after that, by sorting: within one (owner, name), a read takes
⍝ the value of the latest set before it. The owner is the instance GNU APL's dynamic scope
⍝ finds: the nearest up the calls that declares the name. So local names vanish and only global ones
⍝ remain to read: the model's constants, weights and state.
⍝
⍝ The table, one row per operation, in execution order:
⍝   I_F text ('map{' '}map' 'cell' 'ret' besides the flat table's), I_V value id,
⍝   I_A I_B argument value ids, I_X as OP_X (map{ }map: the two ranks; cell: its rank),
⍝   I_I instance, I_R region (0 = none; a region's id is its instance).
⍝ IN_FN holds the function of each instance. IN_BAD lists reads of a local name that
⍝ no set precedes (a compiler error when nonempty).

⍝ ---- rows as seven columns, so they catenate column by column
∇Z←INRowOf A
 Z←(,⊂0⊃A)(,1⊃A)(,2⊃A)(,3⊃A)(,⊂4⊃A)(,5⊃A)(,6⊃A)
∇

⍝ ---- the rows of function F as instance N in region G
∇Z←INCopy A;F;N;G;M;K;X;Q
 (F N G)←A
 M←⍸OP_FN=F
 K←IN_K×N
 X←OP_X[M]
 ⍝ the value ids inside index lists shift too (guarded: ¨ on empty runs on the
 ⍝ prototype in GNU APL)
 Q←⍸OP_F[M]∊⊂'index'
 →(0=⍴Q)/S1
 X[Q]←{⍵+K}¨X[Q]
S1:Q←⍸OP_F[M]∊⊂'setix'
 →(0=⍴Q)/S2
 X[Q]←{(⍵[,0]),(1↓⍵)+K}¨X[Q]
S2:Z←(OP_F[M])(OP_V[M]+K)(OP_A[M]+K×OP_A[M]≥0)(OP_B[M]+K×OP_B[M]≥0)X((⍴M)⍴N)((⍴M)⍴G)
∇

⍝ ---- a mapped function's row in FN_NAME, from text (Name⍤rank); else past the end
∇Z←INMapFn T;C
 Z←↑⍴FN_NAME
 →('('≠↑T)/0
 C←T⍳'⍤'
 →(C≥⍴T)/0
 Z←FN_NAME⍳⊂1↓C↑T
∇

⍝ A new instance of function F
∇Z←INNew F
 Z←⍴IN_FN
 IN_FN←IN_FN,F
 IN_UP←IN_UP,I_I[Q]
∇

⍝ ---- a plain call: set its arguments, the body, read its Z as the call's value
∇Z←INCall Q;F;N;G
 F←C[Q]
 N←INNew F
 G←I_R[Q]
 Z←INCopy F N G
 Z←(INRowOf 'set' ¯2 ¯1 I_B[Q] (F⊃FN_RIGHT) N G),¨Z
 →((I_A[Q]<0)∨0=⍴F⊃FN_LEFT)/DONE
 Z←(INRowOf 'set' ¯2 ¯1 I_A[Q] (F⊃FN_LEFT) N G),¨Z
DONE:Z←Z,¨INRowOf 'name' I_V[Q] ¯1 ¯1 (,'Z') N G
∇

⍝ ---- a map: map{, a cell and a set per argument, the body, read Z, }map. Fresh value
⍝ ids sit at the top of the instance's range, above any node id.
∇Z←INMap Q;F;N;G;T;RK;K;ZV;CV
 F←M[Q]
 N←INNew F
 G←I_R[Q]
 T←Q⊃I_F
 RK←,⍎(1+T⍳'⍤')↓¯1↓T
 RK←¯2↑RK,RK
 K←IN_K×N
 ZV←K+IN_K-1
 Z←INCopy F N N
 CV←K+IN_K-2
 Z←(INRowOf 'set' ¯2 ¯1 CV (F⊃FN_RIGHT) N N),¨Z
 Z←(INRowOf 'cell' CV ¯1 I_B[Q] RK[1] N N),¨Z
 →((I_A[Q]<0)∨0=⍴F⊃FN_LEFT)/BEGIN
 CV←K+IN_K-3
 Z←(INRowOf 'set' ¯2 ¯1 CV (F⊃FN_LEFT) N N),¨Z
 Z←(INRowOf 'cell' CV I_A[Q] ¯1 RK[0] N N),¨Z
BEGIN:Z←(INRowOf 'map{' ¯2 I_A[Q] I_B[Q] RK N G),¨Z
 Z←Z,¨INRowOf 'name' ZV ¯1 ¯1 (,'Z') N N
 Z←Z,¨INRowOf '}map' I_V[Q] ZV I_B[Q] RK N G
∇

∇Z←INRow Q
 Z←(,I_F[Q])(,I_V[Q])(,I_A[Q])(,I_B[Q])(,I_X[Q])(,I_I[Q])(,I_R[Q])
 →(C[Q]<⍴FN_NAME)/CALL
 →(M[Q]<⍴FN_NAME)/MAP
 →0
CALL:Z←INCall Q
 →0
MAP:Z←INMap Q
∇

⍝ ---- one round: every row that calls or maps is replaced by its expansion. The
⍝ result says whether there was any.
∇Z←INRound;C;M;EX
 C←FN_NAME⍳I_F
 M←INMapFn¨I_F
 Z←(∨/C<⍴FN_NAME)∨∨/M<⍴FN_NAME
 →(~Z)/0
 EX←INRow¨⍳⍴I_F
 I_F←↑,/{0⊃⍵}¨EX
 I_V←↑,/{1⊃⍵}¨EX
 I_A←↑,/{2⊃⍵}¨EX
 I_B←↑,/{3⊃⍵}¨EX
 I_X←↑,/{4⊃⍵}¨EX
 I_I←↑,/{5⊃⍵}¨EX
 I_R←↑,/{6⊃⍵}¨EX
∇

⍝ ---- the instance a name belongs to, as dynamic scope finds it: the nearest one up the
⍝ calls whose function declares it, as a local, an argument or Z; ¯1 = global. (The
⍝ entry's own arguments are not local: the step supplies them.)
∇Z←INOwner Q;NAME;F
 NAME←Q⊃I_X
 Z←I_I[Q]
UP:F←IN_FN[Z]
 →((⊂NAME)∊(⊂,'Z'),F⊃FN_LOCALS)/0
 →((Z≠0)∧(⊂NAME)∊(F⊃FN_LEFT)(F⊃FN_RIGHT))/0
 →(Z=0)/GLOBAL
 Z←IN_UP[Z]
 →UP
GLOBAL:Z←¯1
∇

⍝ A value id through the alias table, unchanged if it has no alias
∇Z←INFollow V;P
 P←AL_FROM⍳V
 Z←(V×P=⍴AL_FROM)+(P<⍴AL_FROM)×(AL_TO,0)[P]
∇

⍝ One round of closing the aliases over themselves
∇Z←INClose V
 AL_TO←V
 Z←INFollow V
∇

∇INResolve;NM;OWN;LOC;Q;KEY;ID;O;SET;LAST;P
 NM←I_F∊'name' 'set'
 OWN←NM\INOwner¨NM/⍳⍴I_F
 LOC←NM∧OWN≥0
 Q←⍸LOC
 KEY←(OWN[Q]){(⍕⍺),':',⍵}¨I_X[Q]
 ID←(∪KEY)⍳KEY
 ⍝ the rows of each name together, in table order (⍋ is stable); each row's latest
 ⍝ set is the running maximum of the positions of sets, checked to be its own name's
 O←⍋ID
 Q←Q[O]
 ID←ID[O]
 SET←I_F[Q]∊⊂'set'
 LAST←⌈\SET×1+⍳⍴Q
 IN_BAD←(LAST=0)/Q
 LAST←0⌈LAST-1
 IN_BAD←IN_BAD,(ID≠ID[LAST])/Q
 AL_FROM←I_V[Q]
 AL_TO←I_B[Q[LAST]]
 AL_TO←INClose⍣≡AL_TO
 I_A←INFollow I_A
 I_B←INFollow I_B
 P←⍸I_F∊⊂'index'
 I_X[P]←{INFollow ⍵}¨I_X[P]
 P←⍸I_F∊⊂'setix'
 I_X[P]←{(⍵[,0]),INFollow 1↓⍵}¨I_X[P]
 I_F←(~LOC)/I_F
 I_V←(~LOC)/I_V
 I_A←(~LOC)/I_A
 I_B←(~LOC)/I_B
 I_X←(~LOC)/I_X
 I_I←(~LOC)/I_I
 I_R←(~LOC)/I_R
∇

⍝ ---- one entry function to the table: its rows, then read Z and return it
∇INEntry E;F;ZV
 F←FN_NAME⍳⊂E
 IN_K←4+⌈/OP_V
 IN_FN←,F
 IN_UP←,0
 (I_F I_V I_A I_B I_X I_I I_R)←INCopy F 0 0
 ZV←IN_K-1
 I_F←I_F,(⊂'name'),⊂'ret'
 I_V←I_V,ZV,¯2
 I_A←I_A,¯1 ¯1
 I_B←I_B,¯1,ZV
 I_X←I_X,(⊂,'Z'),⊂''
 I_I←I_I,0 0
 I_R←I_R,0 0
LOOP:→INRound/LOOP
 INResolve
∇
