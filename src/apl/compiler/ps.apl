⍝ ps.apl: the parser of the APL compiler: pairwise reduction after Bunda and Gerth
⍝ (APL two by two, 1984, cited for this purpose by Hsu's ARRAY 2015 paper). All
⍝ tokens of all statements sit in one frontier; each round every adjacent pair
⍝ looks up its binding strength, and every pair stronger than both neighbours
⍝ fuses, all at once: no recursion and no per-statement control.
⍝
⍝ The result is the tree as an inverted table, one row per node (Hsu):
⍝   PN_K kind, PN_L left child, PN_R right child (¯1 = none), PN_T leaf text,
⍝   PN_P 1 if the source put it in parentheses (a strand stops flattening there).
⍝ PSProgram also leaves the statement and function tables: see there.
⍝
⍝ Strengths, strongest first; a higher number binds first. This order is what
⍝ makes right-to-left evaluation come out with no special case: A F (a left
⍝ argument taking its function) outbinds F A, so in x+y×z both x+ and y× form
⍝ before × may take z, and + takes its right argument only when that is finished.
⍝  10 X []  axis or index (tighter than stranding, as in APL)
⍝   9 A A   stranding
⍝   8 F M   monadic operator
⍝   7 F D   left operand of a dyadic operator (also A D)
⍝   6 FD X  its right operand
⍝   5 A ←, F ←
⍝   4 A F   left argument
⍝   3 F A, HF A, AS A   application, assignment
⍝   2 A ;, LP A   index lists
⍝ Triples: ( X ) is X; { X } is a lambda; [ X ] a bracket.
⍝ Blocking pairs have a strength but never fire: a name beside something not yet
⍝ finished must hold back what stands to its left: A [ and F [ (10), A ( (9, a
⍝ strand to come), A FD (4, a function to come), ) F and ] F (4, a left argument
⍝ to come), ) A and ] A (9, a strand to come), and the same for a finished bracket;
⍝ D A and D F (6, an operator's right
⍝ operand: not to be taken by what follows it). Without them Z←M+.×V would take
⍝ Z←M, since M has no partner yet.

⍝ ---- categories
cSEP←0 ⋄ cA←1 ⋄ cF←2 ⋄ cM←3 ⋄ cD←4 ⋄ cFD←5 ⋄ cHF←6 ⋄ cAS←7 ⋄ cL←8 ⋄ cR←9
cLB←10 ⋄ cRB←11 ⋄ cLC←12 ⋄ cRC←13 ⋄ cSC←14 ⋄ cBR←15 ⋄ cG←16 ⋄ cLP←17 ⋄ cLF←18 ⋄ cBAD←19
PS_NC←20

⍝ ---- node kinds
kNAME←0 ⋄ kNUM←1 ⋄ kSTR←2 ⋄ kGLY←3 ⋄ kELIDED←4 ⋄ kSTRAND←5 ⋄ kMON←6 ⋄ kDYA←7 ⋄ kHALF←8
kDERM←9 ⋄ kPART←10 ⋄ kDERD←11 ⋄ kAXIS←12 ⋄ kINDEX←13 ⋄ kBRK←14 ⋄ kASP←15 ⋄ kASG←16
kLAM←17 ⋄ kLIST←18

⍝ ---- the rules: left, right, strength, result category, node kind
PS_RULES←0 5⍴0
PS_RULES←PS_RULES⍪cA,cA,9,cA,kSTRAND
PS_RULES←PS_RULES⍪cF,cM,8,cF,kDERM
PS_RULES←PS_RULES⍪cF,cD,7,cFD,kPART
PS_RULES←PS_RULES⍪cA,cD,7,cFD,kPART
PS_RULES←PS_RULES⍪cFD,cF,6,cF,kDERD
PS_RULES←PS_RULES⍪cFD,cA,6,cF,kDERD
PS_RULES←PS_RULES⍪cF,cBR,10,cF,kAXIS
PS_RULES←PS_RULES⍪cA,cBR,10,cA,kINDEX
PS_RULES←PS_RULES⍪cA,cG,5,cAS,kASP
PS_RULES←PS_RULES⍪cF,cG,5,cAS,kASP
PS_RULES←PS_RULES⍪cA,cF,4,cHF,kHALF
PS_RULES←PS_RULES⍪cF,cA,3,cA,kMON
PS_RULES←PS_RULES⍪cHF,cA,3,cA,kDYA
PS_RULES←PS_RULES⍪cAS,cA,3,cA,kASG
PS_RULES←PS_RULES⍪cAS,cLF,3,cA,kASG
PS_RULES←PS_RULES⍪cA,cSC,2,cLP,kLIST
PS_RULES←PS_RULES⍪cLP,cA,2,cA,kLIST
⍝ the blocking pairs (kind ¯1 never fires)
PS_RULES←PS_RULES⍪cA,cLB,10,0,¯1
PS_RULES←PS_RULES⍪cF,cLB,10,0,¯1
PS_RULES←PS_RULES⍪cA,cL,9,0,¯1
PS_RULES←PS_RULES⍪cA,cFD,4,0,¯1
PS_RULES←PS_RULES⍪cR,cF,4,0,¯1
PS_RULES←PS_RULES⍪cRB,cF,4,0,¯1
PS_RULES←PS_RULES⍪cR,cA,9,0,¯1
PS_RULES←PS_RULES⍪cRB,cA,9,0,¯1
PS_RULES←PS_RULES⍪cBR,cF,4,0,¯1
PS_RULES←PS_RULES⍪cBR,cA,9,0,¯1
PS_RULES←PS_RULES⍪cD,cA,6,0,¯1
PS_RULES←PS_RULES⍪cD,cF,6,0,¯1

⍝ ---- flat tables over a pair (left×PS_NC)+right
PS_S←(PS_NC×PS_NC)⍴0
PS_S[(PS_NC×PS_RULES[;0])+PS_RULES[;1]]←PS_RULES[;2]
PS_C←(PS_NC×PS_NC)⍴0
PS_C[(PS_NC×PS_RULES[;0])+PS_RULES[;1]]←PS_RULES[;3]
PS_K←(PS_NC×PS_NC)⍴0
PS_K[(PS_NC×PS_RULES[;0])+PS_RULES[;1]]←PS_RULES[;4]

⍝ ---- the category of a glyph; the entry past the end is that of anything else.
⍝ ∘ only ever appears as the null function of an outer product here.
PS_GLF←'+-×÷*⍟⌈⌊|○~∧∨=≠<≤>≥⍴⍳,⍪↑↓⊢⊣⌽⍉⍸≢⊃⊂∊⍋⍒⌷∪∘'
PS_GLM←'/⌿\⍀¨⍨'
PS_GLD←'.⍤'
PS_GLA←'⍵⍺⍬'
PS_GLS←'⋄◊'
PS_GLB←'()[]{};←'
PS_GL←PS_GLF,PS_GLM,PS_GLD,PS_GLA,PS_GLS,PS_GLB
PS_GC←((⍴PS_GLF)⍴cF),((⍴PS_GLM)⍴cM),((⍴PS_GLD)⍴cD),((⍴PS_GLA)⍴cA),((⍴PS_GLS)⍴cSEP),cL,cR,cLB,cRB,cLC,cRC,cSC,cG,cBAD

⍝ ---- the initial frontier from tokens K T. Elided index arguments become leaves:
⍝ each [ or ; with ; or ] right after it is doubled and its copy made an elided
⍝ leaf (token kind 5). Names in FNAMES are functions, all other names arrays.
∇PSInit A;K;T;FC;I;M;DUP;CAT
 (K T)←A
 FC←↑¨T
 I←((K=4)∧FC∊'[;')∧1↓((K=4)∧FC∊';]'),0
 M←(1+I)/⍳⍴K
 DUP←~1,2≠/M
 K←K[M]
 T←T[M]
 FC←FC[M]
 K←(K×~DUP)+5×DUP
 T[⍸DUP]←⊂''
 CAT←(cSEP,cA,cA,cA,cBAD,cA)[K]
 CAT←CAT+(K=4)×PS_GC[PS_GL⍳FC]-CAT
 CAT←CAT+((K=1)∧T∊FNAMES)×cF-CAT
 PN_K←(kNAME,kNAME,kNUM,kSTR,kGLY,kELIDED)[K]
 PN_T←T
 PN_L←PN_R←(⍴K)⍴¯1
 PN_P←(⍴K)⍴0
 FR←(2,⍴K)⍴CAT,⍳⍴K
∇

⍝ ---- one round over the whole frontier F (2 rows: categories, node ids)
∇Z←PSStep F;C;I;PIX;S;FIRE;P;PT;LT;BT;Q;KEEP
 C←F[0;]
 I←F[1;]
 PIX←(PS_NC×¯1↓C)+1↓C
 S←PS_S[PIX]
 FIRE←(S>0)∧(S>0,¯1↓S)∧(S≥(1↓S),0)∧PS_K[PIX]≥0
 ⍝ the triples, on the categories before this round's fusions: a bracket pairs
 ⍝ with nothing, and X inside a complete triple has strength 0 on both sides
 PT←(¯2↓C=cL)∧(1↓¯1↓C∊cA,cF)∧2↓C=cR
 LT←(¯2↓C=cLC)∧(1↓¯1↓C=cA)∧2↓C=cRC
 BT←(¯2↓C=cLB)∧(1↓¯1↓C=cA)∧2↓C=cRB
 ⍝ the pairs: a new node per pair, the left slot takes it, the right slot goes
 P←⍸FIRE
 PN_K←PN_K,PS_K[PIX[P]]
 PN_L←PN_L,I[P]
 PN_R←PN_R,I[P+1]
 PN_T←PN_T,(⍴P)⍴⊂''
 PN_P←PN_P,(⍴P)⍴0
 C[P]←PS_C[PIX[P]]
 I[P]←(⍴PN_K)-(⍴P)-⍳⍴P
 KEEP←~0,FIRE
 ⍝ ( X ): X stays, marked
 PN_P[I[1+⍸PT]]←1
 ⍝ { X } and [ X ]: a new node in X's slot
 Q←1+⍸LT
 PN_K←PN_K,(⍴Q)⍴kLAM
 PN_L←PN_L,I[Q]
 PN_R←PN_R,(⍴Q)⍴¯1
 PN_T←PN_T,(⍴Q)⍴⊂''
 PN_P←PN_P,(⍴Q)⍴0
 C[Q]←cLF
 I[Q]←(⍴PN_K)-(⍴Q)-⍳⍴Q
 Q←1+⍸BT
 PN_K←PN_K,(⍴Q)⍴kBRK
 PN_L←PN_L,I[Q]
 PN_R←PN_R,(⍴Q)⍴¯1
 PN_T←PN_T,(⍴Q)⍴⊂''
 PN_P←PN_P,(⍴Q)⍴0
 C[Q]←cBR
 I[Q]←(⍴PN_K)-(⍴Q)-⍳⍴Q
 ⍝ the brackets of every triple go
 KEEP←KEEP∧~((PT∨LT∨BT),0 0)∨0 0,PT∨LT∨BT
 Z←(2,+/KEEP)⍴(KEEP/C),KEEP/I
∇

⍝ ---- one ∇ header, the tokens of its line without the ∇: Z ← [left] name right
⍝ [; local ...]. The result: name, left ('' if monadic), right, locals.
∇Z←PSHeader X;CUT;H
 CUT←X⍳⊂,';'
 H←2↓CUT↑X
 Z←(↑¯2↑H)(↑¯3↑(⊂''),(⊂''),H)(↑¯1↑H)(X[(1+CUT)+⍸~((1+CUT)↓X)∊⊂,';'])
∇

⍝ ---- the program. PSProgram src leaves, besides the node table:
⍝   FN_NAME FN_LEFT FN_RIGHT FN_LOCALS   per ∇ function ('' left if monadic)
⍝   ST_ROOT ST_FN   per statement: its tree, its function (¯1 at top level)
⍝   PS_BAD          frontier positions that did not reduce (empty = a clean parse)
⍝ Header lines are read, not parsed; everything else goes through the reduction.
∇PSProgram S;K;T;FC;LN;DEL;IN;OPEN;BODY;LOPEN;H;FNOF;SEP;NS;FIRST;LAMBDA
 (K T)←TK S
 FC←↑¨T
 LN←¯1↓0,+\K=0
 DEL←(K=4)∧FC='∇'
 IN←2|+\DEL
 OPEN←DEL∧IN
 LOPEN←OPEN/LN
 BODY←~LN∊¯1,DEL/LN
 ⍝ the headers: the tokens of each opening line, without the ∇ and the LF
 H←{((LN=⍵)∧~DEL∨K=0)/T}¨LOPEN
 H←PSHeader¨H
 FN_NAME←{0⊃⍵}¨H
 FN_LEFT←{1⊃⍵}¨H
 FN_RIGHT←{2⊃⍵}¨H
 FN_LOCALS←{3⊃⍵}¨H
 ⍝ the names of lambdas: a name, then ←, then {
 LAMBDA←(K=1)∧(1↓(K=4)∧FC='←'),0
 LAMBDA←LAMBDA∧(2↓(K=4)∧FC='{'),0 0
 FNAMES←FN_NAME,LAMBDA/T
 ⍝ each token's function, ¯1 outside any: the running count of opens where inside
 FNOF←(IN×(+\OPEN)-1)+(~IN)×¯1
 ⍝ each nonempty statement's first body token gives the statement its function
 SEP←(K=0)∨(K=4)∧FC∊'⋄◊'
 NS←BODY∧~SEP
 FIRST←NS∧~¯1↓0,NS
 ST_FN←FIRST/FNOF
 PSInit (BODY/K)(BODY/T)
 FR←PSStep⍣≡FR
 ST_ROOT←(FR[0;]≠cSEP)/FR[1;]
 PS_BAD←⍸~FR[0;]∊cSEP,cA
∇
