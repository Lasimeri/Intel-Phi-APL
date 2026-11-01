⍝ ps_render.apl: render the parser's tree as fully parenthesized APL (TEST ONLY;
⍝ recursion and branches are fine here). If a statement and its rendering evaluate
⍝ to the same value on the same inputs, the parse has the meaning the interpreter
⍝ gives the text: GNU APL is the oracle.
∇Z←R N;K;A;B
 Z←''
 →(N<0)/0
 K←PN_K[N]
 A←PN_L[N]
 B←PN_R[N]
 →(K∊kNAME,kNUM,kSTR,kGLY)/LEAF
 →(K=kELIDED)/0
 →(K=kSTRAND)/STRAND
 →(K=kMON)/MON
 →(K=kDYA)/DYA
 →(K=kHALF)/HALF
 →(K=kDERM)/DERM
 →(K=kPART)/PART
 →(K=kDERD)/DERD
 →(K∊kAXIS,kINDEX)/IDX
 →(K=kBRK)/BRK
 →(K=kASP)/ASP
 →(K=kASG)/ASG
 →(K=kLAM)/LAM
 →(K=kLIST)/LIST
 ⎕ES 'unknown node kind ',⍕K
LEAF:Z←⊃PN_T[N]
 →0
STRAND:Z←'(',(RS N),')'
 →0
MON:Z←'(',(R A),' ',(R B),')'
 →0
DYA:Z←'(',(R PN_L[A]),' ',(R PN_R[A]),' ',(R B),')'
 →0
HALF:Z←'(',(R A),' ',(R B),')'
 →0
DERM:Z←'(',(R A),(R B),')'
 →0
PART:Z←(R A),(R B)
 →0
DERD:Z←'(',(R A),(R B),')'
 →0
IDX:Z←(R A),(R B)
 →0
BRK:Z←'[',(R A),']'
 →0
ASP:Z←(R A),'←'
 →0
ASG:Z←'(',(R A),(R B),')'
 →0
LAM:Z←'{',(R A),'}'
 →0
LIST:Z←(R A),R B
 →0
∇

⍝ A strand's items, flat, unless the source parenthesized the left part
∇Z←RS N
 Z←(RSL PN_L[N]),' ',R PN_R[N]
∇

∇Z←RSL N
 →((PN_K[N]=kSTRAND)∧~PN_P[N])/GO
 Z←R N
 →0
GO:Z←(RSL PN_L[N]),' ',R PN_R[N]
∇

⍝ ---- the whole program back as APL text, every statement fully parenthesized.
⍝ A statement's assignment has no outer parentheses: (Z←X) would print its value.
∇Z←RStmt N
 Z←R N
 →(PN_K[N]≠kASG)/0
 Z←(R PN_L[N]),R PN_R[N]
∇

∇Z←RHeader F;L
 L←F⊃FN_LEFT
 Z←'∇Z←',(L,(0<⍴L)/' '),(F⊃FN_NAME),' ',(F⊃FN_RIGHT),(0<⍴F⊃FN_LOCALS)/∊';',¨F⊃FN_LOCALS
∇

∇Z←RProgram;F;T
 T←(ST_FN=¯1)/ST_ROOT
 Z←RStmt¨T
 F←0
NEXT:→(F≥⍴FN_NAME)/0
 Z←Z,(⊂RHeader F),({' ',RStmt ⍵}¨(ST_FN=F)/ST_ROOT),⊂,'∇'
 F←F+1
 →NEXT
∇

⍝ ---- the flat table back as APL, one operation per line, each value in a variable
⍝ V<id>: the flatten oracle. Text for one row:
∇Z←ROp I;F;V;A;B;X
 F←I⊃OP_F
 V←'V',⍕OP_V[I]
 A←'V',⍕OP_A[I]
 B←'V',⍕OP_B[I]
 X←I⊃OP_X
 Z←' ',V,'←',A,' ',F,' ',B
 →(OP_A[I]≥0)/KIND
 Z←' ',V,'←',F,' ',B
KIND:→(~(⊂F)∊'name' 'num' 'str')/K1
 Z←' ',V,'←',X
 →0
K1:→(~F≡'elided')/K2
 Z←' ',V,'←0'
 →0
K2:→(~F≡'strand')/K3
 Z←' ',V,'←',A,',',B
 →0
K3:→(~F≡'index')/K4
 Z←' ',V,'←',A,'[',(RIx X),']'
 →0
K4:→(~F≡'set')/K5
 Z←' ',V,'←',X,'←',B
 →0
K5:→(~F≡'setix')/0
 Z←' ',(0⊃X),'[',(RIx 1↓X),']←',B
∇

⍝ An index list: the elided items are blank
∇Z←RIx X
 Z←¯1↓∊{((PN_K[⍵]≠kELIDED)/'V',⍕⍵),';'}¨X
∇

⍝ The whole table as APL: each function with its operations, in table order
∇Z←RFlat;F;M
 Z←0⍴⊂''
 F←0
NEXT:→(F≥⍴FN_NAME)/0
 M←⍸OP_FN=F
 Z←Z,(⊂RHeader F),(ROp¨M),⊂,'∇'
 F←F+1
 →NEXT
∇

⍝ ---- the inlined table of one entry back as one APL function: the inliner's oracle.
⍝ A map region becomes a loop with a label. CELL_ takes a cell of an argument.
∇Z←RIRow I;F;V;A;B;X;G
 F←I⊃I_F
 V←'V',⍕I_V[I]
 A←'V',⍕I_A[I]
 B←'V',⍕I_B[I]
 X←I⊃I_X
 Z←,⊂' ',V,'←',A,' ',F,' ',B
 →(I_A[I]≥0)/KIND
 Z←,⊂' ',V,'←',F,' ',B
KIND:→(~(⊂F)∊'name' 'num' 'str')/K1
 Z←,⊂' ',V,'←',X
 →0
K1:→(~F≡'elided')/K2
 Z←,⊂' ',V,'←0'
 →0
K2:→(~F≡'strand')/K3
 Z←,⊂' ',V,'←',A,',',B
 →0
K3:→(~F≡'index')/K4
 Z←,⊂' ',V,'←',A,'[',(RIxI X),']'
 →0
K4:→(~F≡'set')/K5
 Z←,⊂' ',V,'←',X,'←',B
 →0
K5:→(~F≡'setix')/K6
 Z←,⊂' ',(0⊃X),'[',(RIxI 1↓X),']←',B
 →0
K6:→(~F≡'ret')/K7
 Z←,⊂' Z←',B
 →0
K7:→(~F≡'map{')/K8
 G←⍕I_I[I]
 Z←(⊂' IX',G,'←0'),(⊂' OUT',G,'←0⍴⊂0'),(⊂'L',G,':')
 →0
K8:→(~F≡'cell')/K9
 G←⍕I_R[I]
 Z←,⊂' ',V,'←',(⍕X),' CELL_ IX',G,' ',((I_B[I]≥0)⊃A B)
 →0
K9:→(~F≡'}map')/0
 G←⍕I_I[I]
 Z←(⊂' OUT',G,'←OUT',G,',⊂',A),(⊂' IX',G,'←IX',G,'+1'),(⊂' →(IX',G,'<↑⍴',B,')/L',G),⊂' ',V,'←⊃OUT',G
∇

⍝ An index list of the inlined table: the elided items are blank
∇Z←RIxI X
 Z←¯1↓∊{((~((I_V⍳⍵)⊃I_F)≡'elided')/'V',⍕⍵),';'}¨X
∇

∇Z←RInlined E;F;L
 F←FN_NAME⍳⊂E
 L←F⊃FN_LEFT
 Z←(⊂'∇Z←',(L,(0<⍴L)/' '),E,' ',F⊃FN_RIGHT),(↑,/RIRow¨⍳⍴I_F),⊂,'∇'
∇

⍝ ---- the shape oracle: each value row of the inlined program prints its real shape
⍝ as it runs; the compiler writes what it inferred. A row's text gets a print appended.
∇Z←RIRowS I;V
 Z←RIRow I
 V←'V',⍕I_V[I]
 →((⊂I⊃I_F)∊'set' 'setix' 'ret' 'map{' 'elided')/0
 Z←Z,⊂' ⎕←''S ',E,':',(⍕I_V[I]),' '',⍕⍴',V
∇

∇Z←RInlinedS E;F;L
 F←FN_NAME⍳⊂E
 L←F⊃FN_LEFT
 Z←(⊂'∇Z←',(L,(0<⍴L)/' '),E,' ',F⊃FN_RIGHT),((L≡,'p')/⊂' ⎕←''P '',⍕p'),(↑,/RIRowS¨⍳⍴I_F),⊂,'∇'
∇

⍝ What the compiler inferred: one line per value row, entry:id then shape (¯1 = T)
∇Z←E SHLines Y;Q
 Q←⍸~I_F∊'set' 'setix' 'ret' 'map{' 'elided'
 Z←{'S ',E,':',(⍕I_V[⍵]),' ',⍕⍵⊃S_S}¨Q
∇
