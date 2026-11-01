⍝ em.apl: emit C over the runtime (src/card/rt.h) from the inlined, shaped table of one
⍝ entry. EMPlan decides, for the whole table at once, which rows own a place in the
⍝ frame; EMRow then writes each row independently of the others. Every array value is
⍝ V<id>, a pointer; a real scalar s<id>; an integer i<id>; what the shape pass folded is
⍝ a literal or a static table, and writes no code.
⍝
⍝ Fusions, each one measured faster than the plain form: a state's in-place rows, S+a∘.×b into
⍝ one pass, adjacent products into one team round, and map regions the compiler can prove
⍝ independent onto the host's threads. Chains of elementwise rows: fz.apl.
⍝
⍝ Inputs beyond the tables: SH_Q8, the names of the weights in Q8_0; EM_SPLIT, a model split
⍝ with the host; NB, the tokens in a batch. Outputs: E_FRAME,
⍝ the entry's frame bytes; EM_TN EM_TV (float tables) EM_LN EM_LV (integer tables),
⍝ across entries; EM_BAD, what could not be lowered.

EM_EWF←,¨'+-×÷⌈⌊'
EM_EWU←'ADD' 'SUB' 'MUL' 'DIV' 'MAX' 'MIN'
EM_EWL←'add' 'sub' 'mul' 'div' 'max' 'min'
EM_MONF←,¨'-÷*⍟'
EM_MONU←'NEG' 'RCP' 'EXP' 'LOG'

∇EMReset
 EM_TN←0⍴⊂''
 EM_TV←0⍴⊂⍳0
 EM_LN←0⍴⊂''
 EM_LV←0⍴⊂⍳0
 EM_BAD←0⍴⊂''
 EM_FRAME←0
 EM_FH←0⍴⊂''
 EM_COUNTS←0⍴⊂''
∇

⍝ ---- text helpers
∇Z←N EMCall A
 Z←' ',N,'(',(¯2↓∊(⍕¨A),¨⊂', '),');'
∇

⍝ An element count as a C expression; T is the only runtime extent
∇Z←EMCount SZ
 Z←⍕×/(SZ≠¯1)/SZ
 →(~¯1∊SZ)/0
 Z←Z,'*T'
∇

∇Z←EMDim N
 Z←(N=¯1)⊃(⍕N)'T'
∇

⍝ ---- static tables, shared across entries by name
∇Z←EMFloatTable P
 Z←'K',EM_E,'_',⍕I_V[P]
 →((P⊃I_F)≢'name')/ADD
 Z←'K_',P⊃I_X
ADD:→((⊂Z)∊EM_TN)/0
 EM_TN←EM_TN,⊂Z
 EM_TV←EM_TV,⊂,P⊃S_V
∇

∇Z←EMLongTable P
 Z←'KI',EM_E,'_',⍕I_V[P]
 →((P⊃I_F)≢'name')/ADD
 Z←'KI_',P⊃I_X
ADD:→((⊂Z)∊EM_LN)/0
 EM_LN←EM_LN,⊂Z
 EM_LV←EM_LV,⊂,P⊃S_V
∇

⍝ ---- a value as a float-side operand
∇Z←EMOp V;P
 P←SHPos V
 →S_K[P]/KNOWN
 →S_I[P]/INT
 Z←((0=↑⍴P⊃S_S)⊃'Vs'),⍕V
 →0
INT:Z←'i',⍕V
 →0
KNOWN:→(0<↑⍴P⊃S_S)/TABLE
 Z←'((float)',(⍕P⊃S_V),')'
 →0
TABLE:Z←EMFloatTable P
∇

⍝ ---- a scalar integer as an expression
∇Z←EMInt V;P
 P←SHPos V
 Z←'i',⍕V
 →(~S_K[P])/0
 Z←⍕P⊃S_V
∇

⍝ An integer vector's J-th element as an expression; a scalar is itself
∇Z←EMIntEl V;P
 P←SHPos V
 →(0=↑⍴P⊃S_S)/SCALAR
 →S_K[P]/KNOWN
 Z←'i',(⍕V),'[j]'
 →0
KNOWN:Z←(EMLongTable P),'[j]'
 →0
SCALAR:Z←EMInt V
∇

⍝ An index item as (list)(count); N is that axis's length
∇Z←N EMList V;P
 P←SHPos V
 →((P⊃I_F)≡'elided')/ALL
 →(0=↑⍴P⊃S_S)/ONE
 →S_K[P]/KNOWN
 →((P⊃I_F)≡,'⍳')/IOTA
 Z←('i',⍕V)(⍕↑P⊃S_S)
 →0
KNOWN:Z←(EMLongTable P)(⍕↑P⊃S_S)
 →0
IOTA:Z←'NULL'('i',⍕I_B[P])
 →0
ONE:Z←('&(const long){',(EMInt V),'}')(,'1')
 →0
ALL:Z←'NULL'(⍕N)
∇

⍝ ---- does an index row need no buffer? a[i;], a[i;;], a[i;j;;] of a plain array: scalars, then elided
∇Z←EMIsRowView Q;X;A;E
 X←Q⊃I_X
 Z←0
 →(2>⍴X)/0
 A←SHPos I_A[Q]
 →((A⊃I_F)≡'name')/NAMED
 →TEST
NAMED:→((⊂A⊃I_X)∊SH_Q8)/0
TEST:E←I_F[SHPos¨X]∊⊂'elided'
 Z←(~E[0])∧(E≡⌈\E)∧∧/(~E)/{0=↑⍴(SHPos ⍵)⊃S_S}¨X
∇

⍝ The lanes before a row of an array of shape SA, the row's scalar indexes leading X, as C
∇Z←SA EMRowAt X;K;I
 K←+/~I_F[SHPos¨X]∊⊂'elided'
 Z←EMInt X[0]
 I←1
NEXT:→(I≥K)/DONE
 Z←'(',Z,' * ',(⍕SA[I]),' + ',(EMInt X[I]),')'
 I←I+1
 →NEXT
DONE:Z←Z,' * ',⍕×/K↓SA
∇

∇Z←EMIsView Q;F;SRC;SZ
 F←Q⊃I_F
 Z←(⊂F)∊(,'⊢')(,'⍴')(,'↑')(,'↓')
 →Z/0
 ⍝ ravel, and ⍉W: a matrix product takes the weight as it lies
 Z←((F≡,',')∨F≡,'⍉')∧I_A[Q]<0
 →Z/0
 →(F≡'index')/IX
 →(F≡'cell')/CL
 →0
IX:Z←EMIsRowView Q
 →0
CL:SZ←Q⊃S_S
 SRC←(I_A[Q]≥0)⊃I_B[Q],I_A[Q]
 Z←(Q⊃I_X)≥↑⍴(SHPos SRC)⊃S_S
 →Z/0
 Z←(~¯1∊SZ)∧0=16|×/SZ
∇

⍝ A frame place: whole 16-lane blocks; T at its limit
∇Z←EMBytes SZ
 Z←4×16×⌈(×/(SZ×SZ≠¯1)+CTX×SZ=¯1)÷16
∇

⍝ ---- the whole table: who owns what. A state written by an elementwise row takes the
⍝ result in place: the row's inputs are read before each lane is written.
∇EMPlan;N;RANK;ARR;P;K;R;X
 N←⍴I_F
 RANK←{↑⍴⍵}¨S_S
 ARR←(RANK>0)∧(~S_K)∧(~S_I)∧~I_F∊'name' 'set' 'setix' 'ret' 'map{' 'elided' 'strand'
 E_VIEW←EMIsView¨⍳N
 E_DEST←N⍴0
 E_DESTN←N⍴⊂''
 E_DESTI←N⍴0
 E_DESTL←N⍴⊂⍳0
 P←⍸I_F∊⊂'set'
 K←0
NEXT:→(K≥⍴P)/DONE
 R←SHPos I_B[P[K]]
 →(~(⊂R⊃I_F)∊(,'+')'(×⍤(0 2))')/SKIP
 E_DEST[R]←1
 E_DESTN[R]←⊂P[K]⊃I_X
SKIP:K←K+1
 →NEXT
DONE:P←⍸I_F∊⊂'setix'
 K←0
 ⍝ a row of a state written as a×old+b∘.×c, the old row another one: straight into the new
ROWS:→(K≥⍴P)/ROWSDONE
 X←P[K]⊃I_X
 R←SHPos I_B[P[K]]
 →((R⊃I_F)≢,'+')/ROWNEXT
 →((((SHPos I_A[R])⊃I_F)≢,'×')∨((SHPos I_B[R])⊃I_F)≢'(∘.×)')/ROWNEXT
 E_DEST[R]←1
 E_DESTN[R]←X[,0]
 E_DESTI[R]←1
 E_DESTL[R]←⊂1↓X
ROWNEXT:K←K+1
 →ROWS
ROWSDONE:EMFzClear
 E_KV←EMHistory N
 E_FUSED←EMFused N
 E_GQ←N⍴0
 E_BUF←ARR∧(~E_VIEW)∧(~E_DEST)∧(~E_KV)∧~E_FUSED
 EMParamX
 EMParallel
 EMAlloc N
∇

⍝ ---- a×s+b∘.×c into a state needs no array of either term: the sum takes them lane by lane.
⍝ Marks both terms.
∇Z←EMFused N;P;K
 Z←N⍴0
 P←⍸E_DESTI
 K←0
NEXT:→(K≥⍴P)/0
 Z[SHPos¨I_A[P[K]],I_B[P[K]]]←1
 K←K+1
 →NEXT
∇

⍝ ---- 1 0 2⍉KC[⍳T;L;], a head's whole history as an array, is never built: the inner
⍝ products read the state where it is. Marks the index and transpose rows.
∇Z←EMHistory N;P;K;R;X
 Z←N⍴0
 P←⍸I_F∊⊂,'⍉'
 K←0
NEXT:→(K≥⍴P)/THROUGH
 R←SHPos I_B[P[K]]
 →((R⊃I_F)≢'index')/SKIP
 X←R⊃I_X
 →(3≠⍴X)/SKIP
 →((((SHPos X[0])⊃I_F)≢,'⍳')∨S_K[SHPos X[0]])/SKIP
 →(((SHPos X[2])⊃I_F)≢'elided')/SKIP
 Z[P[K],R]←1
SKIP:K←K+1
 →NEXT
THROUGH:P←⍸I_F∊⊂,'⊢'
 K←0
PASS:→(K≥⍴P)/0
 Z[P[K]]←Z[SHPos I_B[P[K]]]
 K←K+1
 →PASS
∇

⍝ The index row under a history view, through any ⊢
∇Z←EMKVIndex V
 Z←SHPos V
NEXT:→((Z⊃I_F)≡'index')/0
 Z←SHPos I_B[Z]
 →NEXT
∇

⍝ ---- an argument of the entry lies outside the frame, where a crossing can't reach it: a
⍝ product on the card that reads one gets it copied into the frame, on the host
∇EMParamX;P;X
 →(~EM_SPLIT)/0
 P←⍸I_F∊⊂'(+.×)'
 →(0=⍴P)/0
 X←SHPos¨EMProdX¨P
 X←(I_F[X]∊⊂'name')/X
 →(0=⍴X)/0
 X←({(⊂⍵⊃I_X)∊0⊃SH_RT}¨X)/X
 →(0=⍴X)/0
 E_BUF[X]←1
∇

⍝ ---- a matrix product, either way round: W+.×x, or x+.×⍉W with x a row per token. The row
⍝ of the weight's name, and the value of the operand.
∇Z←EMProdW Q;B
 B←SHPos I_B[Q]
 Z←SHPos I_A[Q]
 →((B⊃I_F)≢,'⍉')/0
 Z←SHPos I_B[B]
∇

∇Z←EMProdX Q
 Z←I_B[Q]
 →(((SHPos I_B[Q])⊃I_F)≢,'⍉')/0
 Z←I_A[Q]
∇

⍝ What lies under a take of a matrix: its name, for a weight
∇Z←EMBase R
 Z←R
NEXT:→(~(⊂Z⊃I_F)∊(,'↑')(,'⍉'))/0
 Z←SHPos I_B[Z]
 →NEXT
∇

⍝ ---- what a view reads
∇Z←EMViewSource Q
 Z←I_B[Q]
 →((Q⊃I_F)≡'index')/INDEX
 →((Q⊃I_F)≢'cell')/0
 Z←(I_A[Q]≥0)⊃I_B[Q],I_A[Q]
 →0
INDEX:Z←I_A[Q]
∇

⍝ One use, at row Q, of value V. Used inside a map region it was defined before, it
⍝ lives to the region's end: the loop comes back for it.
∇EMUse V;U;AT;G
 →(V<0)/0
 U←BASE[SHPos V]
 AT←(E_FZR[Q]≥0)⊃Q,E_FZR[Q]
 G←I_R[Q]
UP:→(G=0)/SET
 →(DEF[U]≥START[G])/SET
 AT←STOP[G]
 G←I_R[AT]
 →UP
SET:LAST[U]←LAST[U]⌈AT
∇

⍝ Is row Q elementwise? Its result may take the place of an input that dies here.
∇Z←EMIsEW Q
 Z←E_BUF[Q]∧((⊂Q⊃I_F)∊EM_EWF)∨(I_A[Q]<0)∧(⊂Q⊃I_F)∊EM_MONF
∇

⍝ Give back the places of the values whose last use is row Q
∇EMFree Q;R;K
 R←⍸E_BUF∧(LAST=Q)∧HELD∧~OWNED
 K←0
NEXT:→(K≥⍴R)/0
 FO←FO,E_OFF[R[K]]
 FS←FS,EMBytes R[K]⊃S_S
 HELD[R[K]]←0
 K←K+1
 →NEXT
∇

⍝ Take a place of exactly this size, the most recently freed first: the warmest lines
∇EMTake R;B;J;M
 B←EMBytes R⊃S_S
 →(E_PRIV[R]<0)/SHARED
 E_OFF[R]←PSIZE[E_PRIV[R]]
 PSIZE[E_PRIV[R]]←PSIZE[E_PRIV[R]]+B
 →0
SHARED:J←(FS=B)/⍳⍴FS
 →OWNED[R]/NEW
 →(0=⍴J)/NEW
 J←↑¯1↑J
 E_OFF[R]←FO[J]
 M←J≠⍳⍴FO
 FO←M/FO
 FS←M/FS
 →DONE
NEW:E_OFF[R]←TOP
 TOP←TOP+B+OWNED[R]×EM_GAP
DONE:HELD[R]←1
∇

⍝ ---- frame places by last use, one pass over the table: a temporary lands on lines the last
⍝ one just warmed. A matrix product's x and y are the exception: each has a place of its own,
⍝ with EM_GAP bytes after it. Places in contact cost the card about 5 % of its decode (2B,
⍝ 2026-11-01: 46.7 tok/s against 49.1), and reused ones about 8.7 %. A map's result is held
⍝ from the map's opening.
EM_GAP←4096

∇EMAlloc N;BASE;LAST;DEF;START;STOP;Q;P;FO;FS;TOP;HELD;R;K;OWNED;PSIZE
 START←STOP←(⍴IN_FN)⍴0
 DEF←⍳N
 P←⍸I_F∊⊂'map{'
 START[I_I[P]]←P
 P←⍸I_F∊⊂'}map'
 STOP[I_I[P]]←P
 DEF[P]←START[I_I[P]]
 BASE←⍳N
 LAST←DEF
 Q←0
SCAN:→(Q≥N)/WALK0
 →(~E_VIEW[Q]∨E_KV[Q]∧(Q⊃I_F)≡,'⍉')/USES
 BASE[Q]←BASE[SHPos EMViewSource Q]
USES:EMUse I_A[Q]
 EMUse I_B[Q]
 →(I_A[Q]<0)/USEDA
 →(~E_FUSED[SHPos I_A[Q]])/USEDA
 EMUse I_A[SHPos I_A[Q]]
 EMUse I_B[SHPos I_A[Q]]
USEDA:→(I_B[Q]<0)/USED
 →(~E_FUSED[SHPos I_B[Q]])/USED
 EMUse I_A[SHPos I_B[Q]]
 EMUse I_B[SHPos I_B[Q]]
USED:Q←Q+1
 →SCAN
WALK0:P←⍸I_F∊⊂'ret'
 LAST[BASE[SHPos¨I_B[P]]]←N
 E_BASE←BASE
 OWNED←N⍴0
 P←⍸I_F∊⊂'(+.×)'
 →(0=⍴P)/PLACES
 P←({(⊂(EMBase EMProdW ⍵)⊃I_X)∊SH_Q8}¨P)/P
 →(0=⍴P)/PLACES
 OWNED[P]←1
 OWNED[BASE[SHPos¨EMProdX¨P]]←1
PLACES:FO←FS←⍳0
 PSIZE←(⍴E_PARS)⍴0
 TOP←0
 E_OFF←N⍴0
 HELD←N⍴0
 Q←0
WALK:→(Q≥N)/DONE
 →(~EMIsEW Q)/TAKE
 EMFree Q
TAKE:R←⍸E_BUF∧DEF=Q
 K←0
EACH:→(K≥⍴R)/AFTER
 EMTake R[K]
 K←K+1
 →EACH
AFTER:EMFree Q
 Q←Q+1
 →WALK
DONE:E_PARSIZE←PSIZE
 K←0
INNER:→(K≥⍴E_PARS)/ALLOCATED
 R←⍸(E_PRIV=K)∧E_BUF
 E_OFF[R]←E_OFF[R]+TOP
 TOP←TOP+EM_HOST_THREADS×PSIZE[K]
 K←K+1
 →INNER
ALLOCATED:E_FRAME←TOP
∇

⍝ ---- the top of a function: every frame place and in-place result
⍝ (a row of a state is known only where its index is: EMRow declares it)
∇Z←EMDecl Q
 Z←' UNUSED float *V',(⍕I_V[Q]),' = (float *)(F + ',(⍕E_OFF[Q]),');'
 →E_BUF[Q]/0
 Z←' UNUSED float *V',(⍕I_V[Q]),' = L[SLOT_',(Q⊃E_DESTN),'];'
∇

⍝ ---- one row
∇Z←EMRow Q
 Z←EMRowText Q
 →(~E_DESTI[Q])/0
 Z←' float *V',(⍕I_V[Q]),' = L[SLOT_',(Q⊃E_DESTN),'] + ',(((SH_ST⍳E_DESTN[Q])⊃SH_STS) EMRowAt Q⊃E_DESTL),';',TK_LF,Z
∇

∇Z←EMRowText Q;F;A;B;X;V;SZ;SA;SB;P;AN;BN;K;LIST;L0;L1
 F←Q⊃I_F
 A←I_A[Q]
 B←I_B[Q]
 X←Q⊃I_X
 V←⍕I_V[Q]
 SZ←Q⊃S_S
 SA←SB←,¯3
 →(A<0)/R1
 SA←(SHPos A)⊃S_S
R1:→(B<0)/R2
 SB←(SHPos B)⊃S_S
R2:Z←''
 →(E_FZR[Q]≥0)/0
 →(0≠↑⍴Q⊃E_FZS)/FUSE
 →(F≡'map{')/MAPOPEN
 →(F≡'}map')/MAPCLOSE
 →(F≡'ret')/RET
 →(F≡'set')/SET
 →(F≡'setix')/SETIX
 →(F≡'elided')/0
 →E_KV[Q]/0
 →E_FUSED[Q]/0
 →S_K[Q]/0
 →(F≡'name')/NAME
 →(F≡'cell')/CELL
 →(F≡'index')/INDEX
 →S_I[Q]/INTEGER
 →((⊂F)∊(,'⊢')(,'⍴')(,'↑'))/SAME
 →((F≡,',')∧A<0)/SAME
 →(F≡,'↓')/DROP
 →(F≡,',')/CAT
 →(F≡'(+.×)')/GEMV
 →(F≡'(+/)')/SUM
 →(F≡'(⌈/)')/MAXV
 →(F≡'(+⌿)')/REDF
 →(F≡'((+.×)⍤(2 1))')/IP21
 →(F≡'((+.×)⍤(1 2))')/IP12
 →(F≡'((∘.×)⍤(1 1))')/OUTER
 →((⊂F)∊'(×⍤(0 1))' '(×⍤(0 2))')/SCALE
 →(F≡,'⍉')/TRANSPOSE
 →((F≡,'*')∧A≥0)/POWER
 →((A<0)∧(⊂F)∊EM_MONF)/MONAD
 →((⊂F)∊EM_EWF)/DYAD
FUSE:Z←EMFzEmit Q
 →0
BAD:EM_BAD←EM_BAD,⊂EM_E,': ',F,' at V',V
 →0
NAME:→((⊂X)∊0⊃SH_RT)/PARAM
 Z←' UNUSED const float *V',V,' = L[SLOT_',X,'];'
 →0
PARAM:→S_I[Q]/PARAMI
 →E_BUF[Q]/PARAMC
 Z←' UNUSED const float *V',V,' = X;'
 →0
PARAMC:Z←' memcpy(V',V,', X, sizeof(float) * ',(EMCount SZ),');'
 →0
PARAMI:Z←' UNUSED const long ',((0<↑⍴SZ)⊃'' (,'*')),'i',V,' = ',X,';'
 →0
SET:P←SHPos B
 →E_DEST[P]/0
 Z←' memcpy(L[SLOT_',X,'], ',(EMOp B),', sizeof(float) * ',(EMCount SB),');'
 →0
SETIX:→E_DESTI[SHPos B]/0
 K←EMCount 1↓(SH_ST⍳X[,0])⊃SH_STS
 Z←' memcpy(L[SLOT_',(0⊃X),'] + ',(EMInt X[1]),' * ',K,', ',(EMOp B),', sizeof(float) * ',K,');'
 →0
RET:Z←' return ',(EMOp B),';'
 →0
MAPOPEN:Z←' for (long IX',(⍕I_I[Q]),' = 0; IX',(⍕I_I[Q]),' < ',(⍕↑SB),'; IX',(⍕I_I[Q]),'++) {'
 →(I_R[Q]≠0)/0
 ⍝ the last token of a batch: nothing reads the state it writes until the next batch
 Z←Z,TK_LF,' R->last = IX',(⍕I_I[Q]),' == ',(⍕(↑SB)-1),';'
 →0
MAPCLOSE:→(E_FZD[SHPos A]≠Q)/MAPCOPY
 Z←' }'
 →0
MAPCOPY:K←EMCount SA
 Z←' memcpy(V',V,' + IX',(⍕I_I[Q]),' * ',K,', ',(EMOp A),', sizeof(float) * ',K,'); }'
 →0
CELL:P←(A≥0)⊃B,A
 K←(A≥0)⊃SB SA
 →(S_I[Q]∧X<↑⍴K)/CELLI
 BN←EMOp P
 →(X≥↑⍴K)/WHOLE
 K←EMCount SZ
 →E_BUF[Q]/CELLCOPY
 Z←' UNUSED const float *V',V,' = ',BN,' + IX',(⍕I_R[Q]),' * ',K,';'
 →0
CELLCOPY:Z←' memcpy(V',V,', ',BN,' + IX',(⍕I_R[Q]),' * ',K,', sizeof(float) * ',K,');'
 →0
CELLI:Z←' UNUSED const long i',V,' = ',(S_K[SHPos P]⊃('i',⍕P)(EMLongTable SHPos P)),'[IX',(⍕I_R[Q]),'];'
 →0
WHOLE:→S_I[Q]/WHOLEI
 Z←' UNUSED const float *V',V,' = ',BN,';'
 →0
WHOLEI:Z←' UNUSED const long i',V,' = ',BN,';'
 →0
INDEX:AN←EMOp A
 →E_VIEW[Q]/ROWVIEW
 →(3=⍴X)/INDEX3
 →(1=⍴X)/INDEX1
 →(((SHPos A)⊃I_F)≡'name')/INDEXQ
 →INDEXL
INDEXQ:→(~(⊂(SHPos A)⊃I_X)∊SH_Q8)/INDEXL
 Z←'rt_q8_row' EMCall AN (SA[1]) (EMInt X[0]) ('V',V)
 →0
INDEXL:L0←SA[0] EMList X[0]
 Z←'rt_index3' EMCall 'R' ('V',V) AN 1 (SA[1]) (0⊃L0) (1⊃L0) 'NULL' 1
 →0
ROWVIEW:Z←' UNUSED const float *V',V,' = ',AN,' + ',(SA EMRowAt X),';'
 →0
INDEX1:Z←' UNUSED const float s',V,' = ',AN,'[',(EMInt X[0]),'];'
 →0
INDEX3:L0←SA[0] EMList X[0]
 L1←SA[1] EMList X[1]
 →(~((SHPos X[2])⊃I_F)≡'elided')/BAD
 Z←'rt_index3' EMCall 'R' ('V',V) AN (SA[1]) (SA[2]) (0⊃L0) (1⊃L0) (0⊃L1) (1⊃L1)
 →0
INTEGER:→(F≡,'⍳')/IOTAROW
 →(F≡'(∧\)')/ISCAN
 →(F≡'(+/)')/ISUM
 →(0<↑⍴SZ)/INTVEC
 →(F≡,'+')/INTADD
 Z←' UNUSED const long i',V,' = ',(EMInt B),' % ',(EMInt A),';'
 →0
INTADD:Z←' UNUSED const long i',V,' = ',(EMInt A),' + ',(EMInt B),';'
 →0
INTVEC:K←⍕↑SZ
 Z←' long i',V,'[',K,']; for (long j = 0; j < ',K,'; j++) i',V,'[j] = '
 →(F≡,'+')/INTVECADD
 →(F≡,'=')/INTVECEQ
 Z←Z,(EMIntEl B),' % ',(EMIntEl A),';'
 →0
INTVECADD:Z←Z,(EMIntEl A),' + ',(EMIntEl B),';'
 →0
INTVECEQ:Z←Z,(EMIntEl A),' == ',(EMIntEl B),';'
 →0
ISCAN:K←⍕↑SZ
 Z←' long i',V,'[',K,']; for (long j = 0; j < ',K,'; j++) i',V,'[j] = (j ? i',V,'[j - 1] : 1) && i',(⍕B),'[j];'
 →0
ISUM:K←⍕↑SB
 Z←' UNUSED long i',V,' = 0; for (long j = 0; j < ',K,'; j++) i',V,' += i',(⍕B),'[j];'
 →0
IOTAROW:→(A≥0)/INDEXOF
 ⍝ ⍳ of a position: the token in hand's positions so far, every later ¯1 extent
 Z←' T = ',(EMInt B),';'
 →0
INDEXOF:Z←' UNUSED const long i',V,' = rt_index_of(',(EMOp A),', ',(EMCount SA),', ',(EMOp B),');'
 →0
SAME:Z←' UNUSED const float *V',V,' = ',(EMOp B),';'
 →0
DROP:Z←' UNUSED const float *V',V,' = ',(EMOp B),' + ',(⍕|(SHPos A)⊃S_V),';'
 →0
CAT:Z←'rt_cat' EMCall ('V',V) (EMOp A) (EMCount SA) (EMOp B) (EMCount SB)
 →0
GEMV:P←EMProdW Q
 →((⊂(EMBase P)⊃I_X)∊SH_Q8)/GEMVQ
 ⍝ a matrix of the host's own, F32, by a vector
 Z←'rt_ip21' EMCall 'R' ('V',V) (EMOp A) 1 (SA[0]) (SA[1]) (EMOp B)
 →0
GEMVQ:L0←P⊃S_S
 K←EMProdX Q
 L1←(SHPos K)⊃S_S
 →((2=⍴L1)∧1<↑L1)/GEMM
 Z←((E_GQ[Q]>0)⊃'rt_gemv' 'rt_gemv_q') EMCall 'R' (EMOp I_V[P]) (L0[0]) (L0[1]) (EMOp K) ('V',V)
 →(E_GQ[Q]≠2)/0
 Z←Z,TK_LF,' rt_gemv_go(R);'
 →0
GEMM:Z←((E_GQ[Q]>0)⊃'rt_gemm' 'rt_gemm_q') EMCall 'R' (EMOp I_V[P]) (L0[0]) (L0[1]) (EMOp K) (↑L1) ('V',V)
 →(E_GQ[Q]≠2)/0
 Z←Z,TK_LF,' rt_gemv_go(R);'
 →0
SUM:Z←' UNUSED const float s',V,' = rt_sum(',(EMOp B),', ',(EMCount SB),');'
 →0
MAXV:Z←' UNUSED const float s',V,' = rt_maxv(',(EMOp B),', ',(EMCount SB),');'
 →0
REDF:Z←'rt_redfirst' EMCall ('V',V) (EMOp B) (SB[0]) (SB[1])
 →0
IP21:→E_KV[SHPos A]/IP21K
 Z←'rt_ip21' EMCall 'R' ('V',V) (EMOp A) (EMDim SA[0]) (EMDim SA[1]) (EMDim SA[2]) (EMOp B)
 →0
IP21K:P←EMKVIndex A
 L1←((SHPos I_A[P])⊃S_S)[1] EMList (P⊃I_X)[1]
 Z←'rt_ip21_kv' EMCall 'R' ('V',V) (EMOp I_A[P]) (((SHPos I_A[P])⊃S_S)[1]) (((SHPos I_A[P])⊃S_S)[2]) (0⊃L1) (EMDim SA[0]) (EMDim SA[1]) (EMOp B)
 →0
IP12:→E_KV[SHPos B]/IP12K
 Z←'rt_ip12' EMCall 'R' ('V',V) (EMOp A) (EMDim SB[0]) (EMDim SB[1]) (EMDim SB[2]) (EMOp B)
 →0
IP12K:P←EMKVIndex B
 L1←((SHPos I_A[P])⊃S_S)[1] EMList (P⊃I_X)[1]
 Z←'rt_ip12_kv' EMCall 'R' ('V',V) (EMOp A) (EMOp I_A[P]) (((SHPos I_A[P])⊃S_S)[1]) (((SHPos I_A[P])⊃S_S)[2]) (0⊃L1) (EMDim SB[0]) (EMDim SB[1])
 →0
OUTER:Z←'rt_outer11' EMCall 'R' ('V',V) (EMOp A) (EMOp B) (SA[0]) (SA[1]) (SB[1])
 →0
SCALE:Z←'rt_scale0n' EMCall 'R' ('V',V) (EMOp A) (EMOp B) (SA[0]) (×/1↓SB)
 →0
TRANSPOSE:→(A<0)/0
 →(1 0 2≢,(SHPos A)⊃S_V)/BAD
 Z←'rt_transpose102' EMCall 'R' ('V',V) (EMOp B) (EMDim SB[0]) (EMDim SB[1]) (EMDim SB[2])
 →0
POWER:→(~S_K[SHPos B])/BAD
 K←,(SHPos B)⊃S_V
 →(K≡,¯0.5)/RSQRT
 →(K≡,0.5)/SQRT
 →BAD
RSQRT:Z←' UNUSED const float s',V,' = rt_s_rsqrt(',(EMOp A),');'
 →0
SQRT:Z←' UNUSED const float s',V,' = rt_s_sqrt(',(EMOp A),');'
 →0
MONAD:→(0=↑⍴SB)/BAD
 Z←'rt_ew' EMCall 'R' ('RT_',(EM_MONF⍳⊂F)⊃EM_MONU) ('V',V) 'NULL' (EMOp B) 0 (EMCount SB)
 →0
DYAD:→(B<0)/DYAD2
 →(~E_FUSED[SHPos B])/DYAD2
DYAD1:P←SHPos B
 L0←SHPos A
 Z←'rt_scale_outer' EMCall 'R' ('V',V) (EMOp I_B[L0]) (EMOp I_A[L0]) (EMOp I_A[P]) (EMOp I_B[P]) (↑(SHPos I_A[P])⊃S_S) (↑(SHPos I_B[P])⊃S_S)
 →0
DYAD2:K←EM_EWF⍳⊂F
 AN←EMOp A
 BN←EMOp B
 →((0=↑⍴SA)∧0=↑⍴SB)/SCALARS
 →(0=↑⍴SA)/FORMSV
 →(0=↑⍴SB)/FORMVS
 Z←'rt_ew' EMCall 'R' ('RT_',(K⊃EM_EWU),'_VV') ('V',V) AN BN 0 (EMCount SZ)
 →0
FORMSV:Z←'rt_ew' EMCall 'R' ('RT_',(K⊃EM_EWU),'_SV') ('V',V) 'NULL' BN AN (EMCount SZ)
 →0
FORMVS:Z←'rt_ew' EMCall 'R' ('RT_',(K⊃EM_EWU),'_VS') ('V',V) AN 'NULL' BN (EMCount SZ)
 →0
SCALARS:→(~EM_SPLIT)/SCALARC
 →(E_SIDE[Q]≠1)/SCALARC
 →(K>3)/SCALARM
 Z←' UNUSED const float s',V,' = ',AN,' ',(K⊃'+-*/'),' ',BN,';'
 →0
SCALARM:Z←' UNUSED const float s',V,' = ',AN,((K=4)⊃' < ' ' > '),BN,' ? ',AN,' : ',BN,';'
 →0
SCALARC:Z←' UNUSED const float s',V,' = rt_s_',(K⊃EM_EWL),'(',AN,', ',BN,');'
∇

⍝ ---- one entry as C text lines. HEAD: its signature.
∇Z←HEAD EMEntry E;BODY;Q
 EM_E←E,(NB≠1)/⍕NB
 INEntry E
 SH_RT←EMArgs E
 SHEntry
 EM_BAD←EM_BAD,SH_BAD
 →(0=⍴IN_BAD)/PLAN
 EM_BAD←EM_BAD,⊂E,': names read before any set'
PLAN:EMPlan
 →(~EM_SPLIT)/PLANNED
 SCEntry
 EMPlan
PLANNED:→EM_SPLIT/SPLIT
 EM_FRAME←EM_FRAME⌈E_FRAME
 Q←⍸E_BUF∨E_DEST∧~E_DESTI
 BODY←EMRow¨⍳⍴I_F
 BODY←({0<↑⍴⍵}¨BODY)/BODY
 Z←(⊂HEAD),(⊂,'{'),(⊂' UNUSED char *F = R->frame;')
 Z←Z,((⊂,'p')∊0⊃SH_RT)/⊂' UNUSED long T = p + 1;'
 Z←Z,(EMDecl¨Q),BODY,⊂,'}'
 →0
SPLIT:Z←HEAD EMSplit E
∇

⍝ ---- placement: which side runs each row, when the model is split with the host. A row
⍝ is forced by what it is: a matrix product, the state it writes, a map's body. What is
⍝ free goes where a two-state dynamic program over the table finds it cheapest, a change
⍝ of side costing what a crossing measured. No free row is worth a crossing by itself.
EM_SWITCH_NS←7000

∇Z←EMReadsScalar V;P
 Z←0
 →(V<0)/0
 P←SHPos V
 Z←(~S_K[P])∧(~S_I[P])∧0=↑⍴P⊃S_S
∇

∇Z←EMReadsParam V;P
 Z←0
 →(V<0)/0
 P←SHPos V
 →((P⊃I_F)≢'name')/0
 Z←((⊂P⊃I_X)∊0⊃SH_RT)∧~S_I[P]
∇

⍝ 0 the card, 1 the host, 2 both (pointers, integers: no data), ¯1 free. A map's body is the
⍝ host's: its loop has no crossing in it, and the state its tokens take in turn is there.
∇Z←EMForce Q;F;A;B;P
 F←Q⊃I_F
 A←I_A[Q]
 B←I_B[Q]
 Z←1
 →(I_R[Q]≠0)/0
 →((⊂F)∊'map{' '}map' 'ret')/0
 Z←2
 →S_K[Q]/0
 Z←E_BUF[Q]⊃2 1
 →((⊂F)∊'name' 'elided' 'strand' 'num' 'str')/0
 →(E_VIEW[Q]∨E_KV[Q])/0
 →S_I[Q]/INT
 Z←0
 →(F≡'(+.×)')/0
 →(F≡'index')/INDEX
 Z←1
 →((⊂F)∊'set' 'setix')/0
 →(0=↑⍴Q⊃S_S)/0
 →((EMReadsScalar A)∨(EMReadsScalar B)∨(EMReadsParam A)∨EMReadsParam B)/0
 Z←¯1
 →0
INT:Z←1
 →((F≡,'⍳')∧A≥0)/0
 Z←2
 →0
INDEX:P←SHPos A
 Z←¯1
 →((P⊃I_F)≢'name')/0
 Z←0
 →((⊂P⊃I_X)∊SH_Q8)/0
 Z←(⊂P⊃I_X)≡⊂'WIN'
 Z←Z⊃¯1 1
∇

⍝ A free row's cost on each side, ns: (host)(card). Rough constants read off the profile, not a
⍝ fit. From EM_BIG_LANES (rt.c's RT_PAR_LANES) the card runs the row on its team, a round of
⍝ about 12,000 ns, and the host's side pays for the array's copy across, about 1.25 ns a lane
⍝ (3.2 GB/s), which a batch's arrays make the larger part.
EM_BIG_LANES←90000

∇Z←EMCost Q;SZ;N;W;BIG
 SZ←Q⊃S_S
 N←×/(SZ×SZ≠¯1)+128×SZ=¯1
 W←1+3×(⊂Q⊃I_F)∊(,'*')(,'⍟')(,'÷')
 BIG←N≥EM_BIG_LANES
 Z←(30+(0.05×N×W)+BIG×1.25×N),BIG⊃(200+1×N×W)(12000+0.004×N×W)
∇

∇EMPlace;N;FORCE;COST;CH;CC;BH;BC;Q;FH;FC;S
 N←⍴I_F
 FORCE←EMForce¨⍳N
 COST←EMCost¨⍳N
 CH←0
 CC←EM_SWITCH_NS
 BH←BC←N⍴0
 Q←0
WALK:→(Q≥N)/BACK
 →(FORCE[Q]=2)/CARRY
 FH←(FORCE[Q]=0)⊃(0⊃Q⊃COST),1E15
 FC←(FORCE[Q]=1)⊃(1⊃Q⊃COST),1E15
 BH[Q]←(CC+EM_SWITCH_NS)<CH
 BC[Q]←(CH+EM_SWITCH_NS)<CC
 S←FH+CH⌊CC+EM_SWITCH_NS
 CC←FC+CC⌊CH+EM_SWITCH_NS
 CH←S
 →NEXT
CARRY:BH[Q]←BC[Q]←0
NEXT:Q←Q+1
 →WALK
BACK:E_SIDE←N⍴2
 S←1
 Q←N-1
UP:→(Q<0)/DONE
 →(FORCE[Q]=2)/KEEP
 E_SIDE[Q]←S
 S←S≠(S⊃BC[Q],BH[Q])
KEEP:Q←Q-1
 →UP
DONE:E_SWITCHES←+/2≠/(E_SIDE≠2)/E_SIDE
∇

⍝ ---- adjacent products on the card share one team round: a thread runs its share of each
⍝ in turn, with no barrier between. A batch ends at the first row on the card that is not
⍝ a product, at a change of side, or at a product reading an earlier one's result. A batch of
⍝ tokens has no such rounds: there a product takes long enough that a round is nothing to it.
EM_BATCH←8

∇EMBatch;N;Q;K;IDS;CARD
 →(NB>EM_LOGIT_NB)/0
 N←⍴I_F
 CARD←⍸E_SIDE≠2
 K←0
NEXT:→(K≥⍴CARD)/0
 Q←CARD[K]
 →(((Q⊃I_F)≡'(+.×)')∧E_SIDE[Q]=0)/OPEN
 K←K+1
 →NEXT
OPEN:IDS←,Q
MORE:K←K+1
 →(K≥⍴CARD)/CLOSE
 Q←CARD[K]
 →((E_SIDE[Q]≠0)∨(Q⊃I_F)≢'(+.×)')/CLOSE
 →((E_BASE[SHPos EMProdX Q]∊IDS)∨EM_BATCH≤⍴IDS)/CLOSE
 IDS←IDS,Q
 →MORE
CLOSE:E_GQ[IDS]←1
 E_GQ[¯1↑IDS]←2
 →NEXT
∇

⍝ ---- the crossings, from the placement: a row reading a value last written on the other
⍝ side pulls it across at its phase's start. E_XC E_XH: the text each side runs before a
⍝ row.
∇Z←EMXfer B
 Z←(⍕E_OFF[B]),', LINK_BYTES(',(EMCount B⊃S_S),')'
∇

⍝ The copies of a phase's opening (guarded: ¨ on empty leaves a numeric prototype)
∇Z←F EMXfers NEED
 Z←''
 →(0=⍴NEED)/0
 Z←∊{' ',F,'(',(EMXfer ⍵),');',TK_LF}¨NEED
∇

∇EMCross;N;VALID;Q;S;CUR;NEED;START;V;B;K;IN
 N←⍴I_F
 E_XC←E_XH←N⍴⊂''
 VALID←N⍴¯1
 CUR←1
 Q←0
WALK:→(Q≥N)/FINISH
 S←E_SIDE[Q]
 →(S=2)/ROWDONE
 →(S=CUR)/READS
 ⍝ a new phase: its needs, up to the next change of side
 START←Q
 NEED←⍳0
 K←Q
SCAN:→(K≥N)/OPEN
 →(E_SIDE[K]=2)/SKIP
 →(E_SIDE[K]≠S)/OPEN
 IN←I_A[K],I_B[K]
 IN←(IN≥0)/IN
 V←0
EACH:→(V≥⍴IN)/WRITES
 B←E_BASE[SHPos IN[V]]
 →(~E_BUF[B])/NEXTIN
 →(VALID[B]∊S,2)/NEXTIN
 NEED←NEED,B
 VALID[B]←2
NEXTIN:V←V+1
 →EACH
WRITES:→(~E_BUF[K])/SKIP
 VALID[K]←S
SKIP:K←K+1
 →SCAN
OPEN:→(S=0)/TOCARD
 E_XC[START]←⊂'link_push' EMXfers NEED
 E_XC[START]←⊂(START⊃E_XC),' link_card_done();',TK_LF
 E_XH[START]←⊂' link_wait_card();',TK_LF
 →OPENED
TOCARD:E_XH[START]←⊂' link_host_done();',TK_LF
 E_XC[START]←⊂' link_wait_host();',TK_LF,'link_pull' EMXfers NEED
 E_XC[START]←⊂(START⊃E_XC),' link_finish();',TK_LF
OPENED:CUR←S
 →ROWDONE
READS:IN←I_A[Q],I_B[Q]
 IN←(IN≥0)/IN
 V←0
EACH2:→(V≥⍴IN)/WROTE
 B←E_BASE[SHPos IN[V]]
 →(~E_BUF[B])/NEXT2
 →(VALID[B]∊S,2)/NEXT2
 EM_BAD←EM_BAD,⊂EM_E,': value crosses inside a phase at row ',⍕Q
NEXT2:V←V+1
 →EACH2
WROTE:→(~E_BUF[Q])/ROWDONE
 VALID[Q]←S
ROWDONE:Q←Q+1
 →WALK
FINISH:→0
∇

⍝ ---- a map region run by the host's threads, one iteration each: the compiler's proof and its
⍝ code. It is safe when an iteration writes no state but its own row (a setix by the region's
⍝ own cell), and its extents don't depend on a position. It is worth a team round from
⍝ EM_PAR_LANES lanes in all; no row of it may reach that by itself, or the row's own primitive
⍝ would start a round inside the round. Inner temporaries are per thread, EM_HOST_THREADS of
⍝ them. Only a split model does this, and only on the host, where every map's body runs.
EM_PAR_LANES←131072
EM_HOST_THREADS←14

⍝ The rows strictly inside a region opening at S0 and closing at E0
∇Z←S0 EMInside E0
 Z←S0+1+⍳E0-S0+1
∇

∇Z←EMCanSplit A;S0;E0;Q;F;CV;M;LANES
 (S0 E0)←A
 Q←S0 EMInside E0
 F←I_F[Q]
 Z←0
 →(∨/{¯1∊⍵}¨S_S[Q])/0
 →(∨/F∊⊂'set')/0
 →(∨/(F∊⊂,'⍳')∧~S_K[Q])/0
 CV←((F∊⊂'cell')∧I_R[Q]=I_I[S0])/I_V[Q]
 M←⍸F∊⊂'setix'
 →(0=⍴M)/PURE
 →(~∧/{∨/(1↓Q[⍵]⊃I_X)∊CV}¨M)/0
PURE:LANES←(E_BUF∨E_DEST)[Q]×{×/⍵}¨S_S[Q]
 Z←(EM_PAR_LANES≤(↑(SHPos I_B[S0])⊃S_S)×+/LANES)∧EM_PAR_LANES>⌈/LANES
∇

⍝ Which regions run so: E_PARS E_PARE their rows, E_PRIV each row's region or ¯1
∇EMParallel;P;K;S0;E0
 E_PARS←E_PARE←⍳0
 E_PRIV←(⍴I_F)⍴¯1
 →(~EM_SPLIT)/0
 P←⍸I_F∊⊂'map{'
 K←0
NEXT:→(K≥⍴P)/0
 S0←P[K]
 E0←↑⍸(I_F∊⊂'}map')∧I_I=I_I[S0]
 →(E_PRIV[S0]≥0)/SKIP
 →(~EMCanSplit S0,E0)/SKIP
 E_PRIV[S0 EMInside E0]←⍴E_PARS
 E_PARS←E_PARS,S0
 E_PARE←E_PARE,E0
SKIP:K←K+1
 →NEXT
∇

⍝ A captured value's three forms: its field, its copy inside, its value at the call
∇Z←EMCapture P;ID
 ID←⍕I_V[P]
 →S_I[P]/INT
 →(0=↑⍴P⊃S_S)/REAL
 Z←(' float *V',ID,';')(' UNUSED float *V',ID,' = a->V',ID,';')('(float *)V',ID)
 →0
REAL:Z←(' float s',ID,';')(' UNUSED const float s',ID,' = a->s',ID,';')('s',ID)
 →0
INT:→(0<↑⍴P⊃S_S)/INTS
 Z←(' long i',ID,';')(' UNUSED const long i',ID,' = a->i',ID,';')('i',ID)
 →0
INTS:Z←(' const long *i',ID,';')(' UNUSED const long *i',ID,' = a->i',ID,';')('i',ID)
∇

⍝ Region K as a function and its call: (the function's lines)(the call's text)
∇Z←EMOutline K;S0;E0;Q;NAME;CAP;R;INNER;B0;IX;FIELDS;LOCALS;ARGS;A1
 S0←E_PARS[K]
 E0←E_PARE[K]
 Q←(S0 EMInside E0),E0
 NAME←'PM',EM_E,'_',⍕I_I[S0]
 R←SHPos¨∊SCReads¨Q
 R←∪(R,E0)
 R←((R≤S0)∨R≥E0)/R
 R←(~S_K[R])/R
 R←(~I_F[R]∊⊂'elided')/R
 CAP←EMCapture¨R
 FIELDS←{0⊃⍵}¨CAP
 LOCALS←{1⊃⍵}¨CAP
 ARGS←{2⊃⍵}¨CAP
 INNER←((E_PRIV=K)∧E_BUF)/⍳⍴I_F
 B0←⍕↑(SHPos I_B[S0])⊃S_S
 IX←'IX',⍕I_I[S0]
 Z←(⊂'struct ',NAME,'_a { struct rt *R; float *const *L; char *F;'),FIELDS,⊂' };'
 Z←Z,(⊂'static void ',NAME,'(void *arg, UNUSED int tid, UNUSED int n)'),(⊂,'{')
 Z←Z,(⊂' const struct ',NAME,'_a *a = arg;'),(⊂' UNUSED struct rt *R = a->R;'),(⊂' UNUSED float *const *L = a->L;'),⊂' UNUSED char *F = a->F;'
 Z←Z,LOCALS
 Z←Z,{' UNUSED float *V',(⍕I_V[⍵]),' = (float *)(F + ',(⍕E_OFF[⍵]),' + tid * ',(⍕E_PARSIZE[K]),'L);'}¨INNER
 Z←Z,⊂' for (long ',IX,' = ',B0,' * tid / n; ',IX,' < ',B0,' * (tid + 1) / n; ',IX,'++) {'
 Z←Z,TEXT[Q],⊂,'}'
 A1←¯2↓∊ARGS,¨⊂', '
 Z←Z (' struct ',NAME,'_a a',(⍕I_I[S0]),' = {R, L, F, ',A1,'};',TK_LF,' rt_par(R, ',NAME,', &a',(⍕I_I[S0]),');')
∇

⍝ ---- a batch that begins its phase opens its round before the wait for the host: the queue and
⍝ rt_gemv_open go first, and its threads read ahead through the time the card would idle. The
⍝ compiler knows what comes next, so nothing has to find out at run time. Works on TEXT and E_XC.
∇EMOpen;P;K;Q;IDS;L;T;I
 →(NB>EM_LOGIT_NB)/0
 P←⍸E_GQ=2
 K←0
NEXT:→(K≥⍴P)/0
 IDS←,P[K]
 Q←P[K]-1
BACK:→(Q<0)/HAVE
 →(E_SIDE[Q]=2)/STEP
 →(E_GQ[Q]≠1)/HAVE
 IDS←Q,IDS
STEP:Q←Q-1
 →BACK
HAVE:→(~∨/'link_wait_host'⍷IDS[0]⊃E_XC)/SKIP
 L←''
 I←0
LINES:→(I≥⍴IDS)/MOVED
 Q←EMProdW IDS[I]
 Q←EMBase Q
 L←L,(Q⊃TEXT),TK_LF
 TEXT[Q]←⊂''
 Q←EMProdW IDS[I]
 →((Q⊃I_F)≡'name')/ONE
 L←L,(Q⊃TEXT),TK_LF
 TEXT[Q]←⊂''
ONE:T←IDS[I]⊃TEXT
 L←L,((T⍳↑TK_LF)↑T),TK_LF
 TEXT[IDS[I]]←⊂''
 I←I+1
 →LINES
MOVED:E_XC[IDS[0]]←⊂L,' rt_gemv_open(R);',TK_LF,IDS[0]⊃E_XC
 TEXT[P[K]]←⊂' rt_gemv_go(R);'
SKIP:K←K+1
 →NEXT
∇

⍝ ---- one entry split in two: Z is the card's lines; the host's join EM_FH
∇Z←HEAD EMSplit E;N;TEXT;C;H;DECL;TOP;OUT;K;P2
 N←⍴I_F
 EMPlace
 EMFzPlan
 EMAlloc N
 EM_FRAME←EM_FRAME⌈E_FRAME
 E_FZFN←0⍴⊂''
 EMBatch
 EMCross
 TEXT←EMRow¨⍳N
 EMOpen
 C←E_XC,¨(E_SIDE≠1)/¨TEXT
 H←E_XH,¨(E_SIDE≠0)/¨TEXT
 OUT←E_FZFN
 K←0
PAR:→(K≥⍴E_PARS)/PARDONE
 P2←EMOutline K
 OUT←OUT,0⊃P2
 H[((E_PARS[K]) EMInside E_PARE[K]),E_PARE[K]]←⊂''
 H[E_PARS[K]]←⊂(E_PARS[K]⊃E_XH),1⊃P2
 K←K+1
 →PAR
PARDONE:
 C←({0<↑⍴⍵}¨C)/C
 H←({0<↑⍴⍵}¨H)/H
 DECL←EMDecl¨⍸(E_PRIV<0)∧E_BUF∨E_DEST∧~E_DESTI
 TOP←(⊂HEAD),(⊂,'{'),(⊂' UNUSED char *F = R->frame;')
 TOP←TOP,((⊂,'p')∊0⊃SH_RT)/⊂' UNUSED long T = p + 1;'
 Z←TOP,DECL,C,(⊂' return 0;'),⊂,'}'
 EM_FH←EM_FH,OUT,TOP,DECL,H,(⊂' return 0;'),⊂,'}'
 EM_COUNTS←EM_COUNTS,⊂EM_E (+/E_SIDE=0) (+/E_SIDE=1) (E_SWITCHES)
∇
