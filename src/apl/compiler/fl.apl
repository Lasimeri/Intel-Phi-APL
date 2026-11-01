⍝ fl.apl: flatten: from the parser's tree (ps.apl) to one linear table of operations
⍝ in execution order. Whole-array passes: subtree sizes, execution offsets, statement
⍝ membership and function membership are fixed points over the child vectors. A
⍝ function's meaning is the canonical text of its tree, so the compiler's whitelist of
⍝ primitives can be a table keyed on strings like ((+.×)⍤(2 1)).
⍝
⍝ FLProgram leaves, one row per operation:
⍝   OP_F  'name' 'num' 'str' 'elided' 'strand' 'index' 'set' 'setix', or the canonical
⍝         text of the function it applies
⍝   OP_V  the value it produces (its node id)
⍝   OP_A  left argument, a value id or ¯1
⍝   OP_B  right argument, a value id or ¯1
⍝   OP_X  name: the name; num, str: the text; index: the index value ids (¯1 elided);
⍝         set: the name; setix: the name, then the index value ids
⍝   OP_FN the function it belongs to, a row of FN_NAME
⍝ A lambda Name←{e} becomes a function of (⍺, ⍵) whose one statement is Z←e, so all
⍝ functions have one shape: arguments, statements, result Z.
⍝ Top-level statements other than lambdas are rejected: the unit defines functions.

∇FLProgram;N;R0;L0;SZ;ROOT;LAMS;NAMES;F0;ST;O;POS;SKIP;ASP;TG;I;KK;M;EXPR;LIVE
 N←⍴PN_K
 L0←PN_L+(N+1)×PN_L<0
 R0←PN_R+(N+1)×PN_R<0
 ⍝ subtree sizes: children have smaller ids, so the fixed point settles by depth
 SZ←{1+(⍵,0)[L0]+(⍵,0)[R0]}⍣≡N⍴1
 ⍝ execution offsets: right child first, then left, then the node (APL's order)
 O←N⍴0
 O[ST_ROOT]←0,¯1↓+\SZ[ST_ROOT]
 O←{FLDown ⍵}⍣≡O
 POS←O+SZ-1
 ⍝ each node's statement, handed down from its root
 ST←N⍴0
 ST[ST_ROOT]←⍳⍴ST_ROOT
 ST←{FLKids ⍵}⍣≡ST
 ⍝ lambdas become functions (⍺, ⍵): Z←e. Their statements get the new function rows.
 LAMS←(PN_K[PN_R[ST_ROOT×PN_K[ST_ROOT]=kASG]]=kLAM)∧PN_K[ST_ROOT]=kASG
 ROOT←LAMS/ST_ROOT
 F0←⍴FN_NAME
 NAMES←PN_T[PN_L[PN_L[ROOT]]]
 FN_NAME←FN_NAME,NAMES
 FN_LEFT←FN_LEFT,(⍴ROOT)⍴⊂,'⍺'
 FN_RIGHT←FN_RIGHT,(⍴ROOT)⍴⊂,'⍵'
 FN_LOCALS←FN_LOCALS,(⍴ROOT)⍴⊂0⍴⊂''
 ST_FN[⍸LAMS]←F0+⍳⍴ROOT
 FL_BAD←(ST_FN<0)/ST_ROOT
 ⍝ function parts: below the function of an application, downward through derived
 ⍝ functions, axes and index lists
 FL_ISF←N⍴0
 M←⍸PN_K=kMON
 FL_ISF[PN_L[M]]←1
 M←⍸PN_K=kHALF
 FL_ISF[PN_R[M]]←1
 FL_ISF←{FLFnDown ⍵}⍣≡FL_ISF
 FLTexts
 ⍝ what never becomes an operation: function parts and bookkeeping nodes; the name
 ⍝ and arrow of an assignment; the target of an indexed assignment; list separators
 SKIP←FL_ISF∨PN_K∊kHALF,kASP,kBRK,kLIST,kLAM
 ASP←⍸PN_K=kASP
 SKIP[PN_L[ASP]]←1
 SKIP[PN_R[ASP]]←1
 TG←(PN_K[PN_L[ASP]]=kINDEX)/PN_L[ASP]
 SKIP[PN_L[TG]]←1
 M←⍸PN_K=kLIST
 SKIP[(PN_K[PN_R[M]]=kGLY)/PN_R[M]]←1
 ⍝ only what a statement root reaches: separators and brackets left the frontier
 ⍝ as orphan leaves
 LIVE←N⍴0
 LIVE[ST_ROOT]←1
 LIVE←{FLFnDown ⍵}⍣≡LIVE
 I←((~SKIP)∧LIVE)/⍳N
 I←I[⍋POS[I]]
 OP_V←I
 KK←PN_K[I]
 OP_F←OP_X←(⍴I)⍴⊂''
 OP_A←OP_B←(⍴I)⍴¯1
 OP_FN←ST_FN[ST[I]]
 M←⍸KK∊kNAME,kGLY
 OP_F[M]←⊂'name'
 OP_X[M]←PN_T[I[M]]
 M←⍸KK=kNUM
 OP_F[M]←⊂'num'
 OP_X[M]←PN_T[I[M]]
 M←⍸KK=kSTR
 OP_F[M]←⊂'str'
 OP_X[M]←PN_T[I[M]]
 M←⍸KK=kELIDED
 OP_F[M]←⊂'elided'
 M←⍸KK=kSTRAND
 OP_F[M]←⊂'strand'
 OP_A[M]←PN_L[I[M]]
 OP_B[M]←PN_R[I[M]]
 M←⍸KK=kMON
 OP_F[M]←FL_T[PN_L[I[M]]]
 OP_B[M]←PN_R[I[M]]
 M←⍸KK=kDYA
 OP_F[M]←FL_T[PN_R[PN_L[I[M]]]]
 OP_A[M]←PN_L[PN_L[I[M]]]
 OP_B[M]←PN_R[I[M]]
 M←⍸KK=kINDEX
 OP_F[M]←⊂'index'
 OP_A[M]←PN_L[I[M]]
 OP_X[M]←FLItems¨PN_R[I[M]]
 ⍝ assignments, plain and indexed; a lambda's root sets Z from its expression
 M←⍸KK=kASG
 OP_F[M]←⊂'set'
 EXPR←PN_R[I[M]]
 OP_B[M]←EXPR+(PN_K[EXPR]=kLAM)×PN_L[EXPR×PN_K[EXPR]=kLAM]-EXPR
 TG←PN_L[PN_L[I[M]]]
 OP_X[M]←PN_T[TG]
 OP_X[(PN_K[EXPR]=kLAM)/M]←⊂,'Z'
 M←(PN_K[TG]=kINDEX)/M
 TG←(PN_K[TG]=kINDEX)/TG
 OP_F[M]←⊂'setix'
 OP_X[M]←{(PN_T[PN_L[⍵]]),FLItems PN_R[⍵]}¨TG
∇

⍝ One round: a node hands its offset down: the right child starts where it starts,
⍝ the left child after the right child's subtree.
∇Z←FLDown O;KR;KL
 Z←O
 KR←⍸PN_R≥0
 Z[PN_R[KR]]←O[KR]
 KL←⍸PN_L≥0
 Z[PN_L[KL]]←O[KL]+(SZ,0)[R0[KL]]
∇

⍝ One round: children take their parent's value
∇Z←FLKids V;K
 Z←V
 K←⍸PN_L≥0
 Z[PN_L[K]]←V[K]
 K←⍸PN_R≥0
 Z[PN_R[K]]←V[K]
∇

⍝ One round: below a function part, everything is a function part
∇Z←FLFnDown F;K
 Z←F
 K←⍸F∧PN_L≥0
 Z[PN_L[K]]←1
 K←⍸F∧PN_R≥0
 Z[PN_R[K]]←1
∇

⍝ The canonical text of each function part, children first (ids ascending)
∇FLTexts;K;A;B;TA;TB;Q
 FL_T←(⍴PN_K)⍴⊂''
 K←⍸FL_ISF
 Q←0
NEXT:→(Q≥⍴K)/0
 FL_T[K[Q]]←⊂FLText K[Q]
 Q←Q+1
 →NEXT
∇

∇Z←FLText N;K;A;B;TA;TB
 K←PN_K[N]
 A←PN_L[N]
 B←PN_R[N]
 TA←TB←''
 →(A<0)/T1
 TA←A⊃FL_T
T1:→(B<0)/T2
 TB←B⊃FL_T
T2:Z←N⊃PN_T
 →(K∊kNAME,kNUM,kSTR,kGLY)/0
 Z←TA,TB
 →(K∊kPART,kAXIS)/0
 Z←'(',TA,TB,')'
 →(K∊kDERM,kDERD)/0
 Z←'(',TA,' ',TB,')'
 →(K=kSTRAND)/0
 Z←'[',TA,']'
∇

⍝ The value ids in an index bracket, left to right. A list can nest either way (two
⍝ ; pairs may fire in one round), so a fixed point opens every list node in place
⍝ into its two children; the ; leaves then go.
∇Z←FLItems B;SEMI
 Z←{FLOpen ⍵}⍣≡,PN_L[B]
 SEMI←(PN_K[Z]=kGLY)∧{';'=↑⍵⊃PN_T}¨Z
 Z←(~SEMI)/Z
∇

∇Z←FLOpen V;M;I;SECOND
 M←PN_K[V]=kLIST
 I←(1+M)/⍳⍴V
 SECOND←I=¯1↓¯1,I
 Z←V[I]
 Z[⍸M[I]∧~SECOND]←PN_L[Z[⍸M[I]∧~SECOND]]
 Z[⍸SECOND]←PN_R[Z[⍸SECOND]]
∇
