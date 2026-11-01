⍝ sh.apl: shapes and constant folding over the inlined table (in.apl), one pass in
⍝ execution order. The compiler being APL, folding a constant means executing it: an
⍝ operation whose inputs are all known runs here, in the interpreter, on the actual
⍝ values. Everything else gets a shape from a rule table keyed on the function's
⍝ canonical text. That table is the compiler's whitelist of primitives: a text with no
⍝ rule is rejected, with its text.
⍝
⍝ The only runtime extent is T, the number of positions so far of the token in hand: it appears in a
⍝ shape as ¯1. p itself is a runtime integer: arithmetic on it stays integer.
⍝
⍝ Every name must be declared by the driver, because this pass runs inside the
⍝ compiler's own workspace and APL's scoping is dynamic: evaluating an undeclared name
⍝ would read whatever variable of the compiler has that name. Vectors of names, each
⍝ name a character vector:
⍝   SH_RT   this entry's runtime arguments: (names)(shapes)
⍝   SH_RI   which runtime names are integers
⍝   SH_W    weights, shapes in SH_WS
⍝   SH_ST   state, shapes in SH_STS: the only names the program may write
⍝   SH_C    model constants: read by executing the name
⍝
⍝ SHEntry leaves, per row of the table:
⍝   S_K  1 if the value is known at compile time
⍝   S_V  the value, when known
⍝   S_S  the shape (a vector; ¯1 = T, the current token's positions so far)
⍝   S_I  1 if a runtime integer (p and what is computed from it)
⍝   S_E  1 if an elided index item
⍝ and SH_BAD, the texts of what was rejected.

SH_SD←,¨'+-×÷*⌈⌊|'
SH_SM←,¨'-÷*⍟⊢'

⍝ ---- a scalar function's shape: equal shapes, or one side a single element
∇Z←A SHScalar B
 Z←A
 →(A≡B)/0
 →((×/B)=1)/0
 Z←B
 →((×/A)=1)/0
 Z←¯2
∇

⍝ ---- the shape of a row whose inputs are not all known; ¯2 = no rule. The right
⍝ argument: the two shapes (,¯3 = no left argument), and whether the left is known and
⍝ its value, for the structural functions whose shape it decides.
∇Z←F SHRule A;SA;SB;KL;LV
 (SA SB KL LV)←A
 Z←¯2
 →((SA≡,¯3)∧(⊂F)∊SH_SM)/MONAD
 →((⊂F)∊SH_SD)/DYAD
 →((⊂F)∊'(+/)' '(⌈/)')/REDLAST
 →(F≡'(+⌿)')/REDFIRST
 →((F≡,'⍉')∧SA≡,¯3)/REVERSE
 →((F≡,'⍴')∧KL)/RESHAPE
 →((F≡,'↑')∧KL)/TAKE
 →((F≡,'↓')∧KL)/DROP
 →((F≡,'⍉')∧KL)/TRANSPOSE
 →((F≡,',')∧SA≡,¯3)/RAVEL
 →(F≡,',')/CATENATE
 →(F≡'(+.×)')/INNER
 →(F≡'((+.×)⍤(2 1))')/INNER21
 →(F≡'((+.×)⍤(1 2))')/INNER12
 →(F≡'((∘.×)⍤(1 1))')/OUTER11
 →(F≡'(∘.×)')/OUTER
 →((⊂F)∊'(×⍤(0 1))' '(×⍤(0 2))')/SCALEROWS
 →0
MONAD:Z←SB
 →0
DYAD:Z←SA SHScalar SB
 →0
REDLAST:Z←¯1↓SB
 →0
REDFIRST:Z←1↓SB
 →0
RESHAPE:Z←,LV
 →0
TAKE:Z←|,LV
 →0
DROP:Z←SB-|,LV
 →0
REVERSE:Z←⌽SB
 →0
TRANSPOSE:Z←SB[⍋LV]
 →0
RAVEL:Z←,×/SB
 →0
CATENATE:Z←,(↑SA)+↑SB
 →0
INNER:Z←(¯1↓SA),1↓SB
 →0
INNER21:Z←SA[0 1]
 →0
INNER12:Z←SA[,0],SB[,2]
 →0
OUTER:Z←SA,SB
 →0
OUTER11:Z←SA,SB[,1]
 →0
SCALEROWS:Z←SB
∇

⍝ The row of a value id
∇Z←SHPos V
 Z←I_V⍳V
∇

⍝ Is this value a runtime integer? A missing argument (¯1) reads 0.
∇Z←SHInt V
 Z←(S_I,0)[I_V⍳V]
∇

⍝ ---- an index's shape: an elided axis keeps its length, a scalar index drops the
⍝ axis, a vector index takes the index's length
∇Z←SHIndex A;SX;IT;K;P
 (SX IT)←A
 Z←⍳0
 K←0
NEXT:→(K≥⍴IT)/0
 P←SHPos IT[K]
 Z←Z,S_E[P]⊃(P⊃S_S)SX[K]
 K←K+1
 →NEXT
∇

⍝ ---- a fully known index, as text to execute
∇Z←SHIxText X
 Z←¯1↓∊{((~S_E[SHPos ⍵])/'((SHPos ',(⍕⍵),')⊃S_V)'),';'}¨X
∇

⍝ ---- one row: sets S_K S_V S_S S_I S_E at Q
∇SHRow Q;F;A;B;X;KA;KB;SA;SB;LV;LHS;RHS;R
 F←Q⊃I_F
 A←I_A[Q]
 B←I_B[Q]
 X←Q⊃I_X
 KA←KB←1
 SA←SB←,¯3
 LV←0
 →(A<0)/NOLEFT
 KA←S_K[SHPos A]
 SA←(SHPos A)⊃S_S
 LV←(SHPos A)⊃S_V
NOLEFT:→(B<0)/NORIGHT
 KB←S_K[SHPos B]
 SB←(SHPos B)⊃S_S
NORIGHT:→(F≡'name')/NAME
 →((⊂F)∊'num' 'str')/LITERAL
 →(F≡'elided')/ELIDED
 →(F≡'strand')/STRAND
 →(F≡'index')/INDEX
 →(F≡'set')/SET
 →((⊂F)∊'setix' 'ret' 'map{')/0
 →(F≡'cell')/CELL
 →(F≡'}map')/UNMAP
 ⍝ an applied function. The first axis's length is a constant unless it is T.
 →((F≡,'≢')∧¯1≠↑SB)/TALLY
 →(KA∧KB)/EXECUTE
 →(((⊂F)∊,¨'+|-=')∧(KA∨SHInt A)∧KB∨SHInt B)/INTEGER
 →((F≡,'⍳')∧(A<0)∧SHInt B)/IOTA
 →((F≡,'⍳')∧A≥0)/INDEXOF
 →(((⊂F)∊'(∧\)' '(+/)')∧SHInt B)/INTRED
 R←F SHRule SA SB KA LV
 →(R≡¯2)/REJECT
 S_S[Q]←⊂,R
 →0
TALLY:R←↑SB
 →KNOWN
INTRED:S_I[Q]←1
 S_S[Q]←⊂(F≡'(∧\)')⊃(⍳0) SB
 →0
INTEGER:S_I[Q]←1
 S_S[Q]←⊂SA SHScalar SB
 →0
IOTA:S_I[Q]←1
 S_S[Q]←⊂,¯1
 →0
INDEXOF:S_I[Q]←1
 S_S[Q]←⊂SB
 →0
REJECT:SH_BAD←SH_BAD,⊂F
 →0
EXECUTE:RHS←(SHPos B)⊃S_V
 →(A<0)/MONADIC
 LHS←LV
 R←⍎'LHS ',F,' RHS'
 →KNOWN
MONADIC:R←⍎F,' RHS'
KNOWN:S_K[Q]←1
 S_V[Q]←⊂R
 S_S[Q]←⊂⍴R
 →0
NAME:→((⊂X)∊0⊃SH_RT)/RUNTIME
 →((⊂X)∊SH_W)/WEIGHT
 →((⊂X)∊SH_ST)/STATE
 →((⊂X)∊SH_C)/CONSTANT
 SH_BAD←SH_BAD,⊂'undeclared name ',X
 →0
RUNTIME:S_S[Q]←⊂((0⊃SH_RT)⍳⊂X)⊃1⊃SH_RT
 S_I[Q]←(⊂X)∊SH_RI
 →0
WEIGHT:S_S[Q]←SH_WS[SH_W⍳⊂X]
 →0
STATE:S_S[Q]←SH_STS[SH_ST⍳⊂X]
 →0
CONSTANT:R←⍎X
 →KNOWN
LITERAL:R←⍎X
 →KNOWN
ELIDED:S_E[Q]←1
 S_K[Q]←1
 →0
STRAND:→(~KA∧KB)/REJECT
 R←LV,(SHPos B)⊃S_V
 →KNOWN
INDEX:S_S[Q]←⊂SHIndex SA X
 S_I[Q]←SHInt A
 →(~KA∧∧/S_K[SHPos¨X])/0
 R←⍎'LV[',(SHIxText X),']'
 →KNOWN
SET:→((⊂X)∊SH_ST)/0
 SH_BAD←SH_BAD,⊂'write to a non-state name ',X
 →0
CELL:→(A<0)/CELLRIGHT
 S_S[Q]←⊂(-X⌊⍴SA)↑SA
 S_I[Q]←SHInt A
 →(~KA∧X≥⍴SA)/0
 S_K[Q]←1
 S_V[Q]←⊂LV
 →0
CELLRIGHT:S_S[Q]←⊂(-X⌊⍴SB)↑SB
 ⍝ an item of a known vector under rank 0 is an integer of the loop: ⍳NB's
 S_I[Q]←(SHInt B)∨KB∧X=0
 →0
UNMAP:S_S[Q]←⊂(1↑SB),SA
∇

⍝ ---- one entry: the rows in order
∇SHEntry;N;Q
 N←⍴I_F
 S_K←S_I←S_E←N⍴0
 S_V←N⍴⊂0
 S_S←N⍴⊂⍳0
 SH_BAD←0⍴⊂''
 Q←0
NEXT:→(Q≥N)/0
 SHRow Q
 Q←Q+1
 →NEXT
∇
