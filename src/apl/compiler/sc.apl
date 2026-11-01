⍝ sc.apl: schedule. The table's order is the source's order, and that order is not the data's:
⍝ WIN[4|p;]←W_QKV+.×H puts a write on the host between products on the card that all read
⍝ only H. Each change of side costs a crossing both ways, so within what the data allow the
⍝ rows of one side are gathered into as few phases as possible, a list schedule: take every
⍝ ready row of the current side, and change sides only when none is left.
⍝
⍝ What orders two rows: a value one reads from the other, and a state: every row that reads
⍝ or writes a state keeps its place among the others that do. A map region moves whole. The
⍝ return comes last. Rows of pointers are taken as soon as they are ready. A row free to go
⍝ either way counts as the host's, where placement nearly always puts it, and placement then
⍝ decides it for real.
⍝
⍝ SCEntry permutes the table's columns in place. It needs the plan (E_VIEW E_KV E_DEST) and
⍝ the forced sides (EMForce), from before the permutation; the plan is redone after it.

⍝ The value ids a row reads
∇Z←SCReads Q;F;X
 Z←I_A[Q],I_B[Q]
 F←Q⊃I_F
 X←Q⊃I_X
 →(F≡'index')/LIST
 →(F≢'setix')/DONE
 X←1↓X
LIST:Z←Z,X
DONE:Z←(Z≥0)/Z
∇

⍝ The state a pointer row points into, by the state's index in SH_ST; ¯1 when none
∇Z←SCPointerTag Q;F;X
 F←Q⊃I_F
 X←Q⊃I_X
 Z←¯1
 →(F≢'name')/THROUGH
 →(~(⊂X)∊SH_ST)/0
 Z←SH_ST⍳⊂X
 →0
THROUGH:→(~E_VIEW[Q]∨E_KV[Q])/0
 Z←TAG[SHPos EMViewSource Q]
∇

⍝ The states a working row reads or writes
∇Z←SCTouches Q;F;R
 F←Q⊃I_F
 R←SCReads Q
 Z←⍳0
 →(0=⍴R)/OWN
 Z←TAG[SHPos¨R]
OWN:→(F≡'set')/SETS
 →(F≢'setix')/DONE
 Z←Z,SH_ST⍳(Q⊃I_X)[,0]
 →DONE
SETS:Z←Z,SH_ST⍳⊂Q⊃I_X
DONE:Z←∪(Z≥0)/Z
∇

∇SCEntry;N;NODE;TAG;DEP;SIDE;PREV;DONE;HEADS;ORD;CUR;Q;P;K;G;R;T;LEFT;READY;S;M
 N←⍴I_F
 ⍝ a map region is one node, named by its opening row
 NODE←⍳N
 ⍝ the outermost regions: a nested one moves with them
 P←⍸(I_F∊⊂'map{')∧I_R=0
 K←0
BLOCK:→(K≥⍴P)/TAGS
 R←↑⍸(I_F∊⊂'}map')∧I_I=I_I[P[K]]
 NODE[P[K]+⍳1+R-P[K]]←P[K]
 K←K+1
 →BLOCK
TAGS:TAG←N⍴¯1
 DEP←N⍴⊂⍳0
 PREV←(⍴SH_ST)⍴¯1
 Q←0
WALK:→(Q≥N)/ENDS
 TAG[Q]←SCPointerTag Q
 R←SCReads Q
 →(0=⍴R)/STATES
 DEP[NODE[Q]]←⊂(NODE[Q]⊃DEP),NODE[SHPos¨R]
STATES:→(E_VIEW[Q]∨E_KV[Q]∨(Q⊃I_F)≡'name')/NEXT
 T←SCTouches Q
 K←0
EACH:→(K≥⍴T)/NEXT
 →(PREV[T[K]]<0)/FIRST
 DEP[NODE[Q]]←⊂(NODE[Q]⊃DEP),NODE[PREV[T[K]]]
FIRST:PREV[T[K]]←Q
 K←K+1
 →EACH
NEXT:Q←Q+1
 →WALK
 ⍝ the return last
ENDS:R←⍸I_F∊⊂'ret'
 DEP[R]←⊂NODE
 SIDE←EMForce¨⍳N
 SIDE←SIDE+2×SIDE=¯1
 DONE←N⍴0
 HEADS←(NODE=⍳N)/⍳N
 ORD←⍳0
 CUR←1
TAKE:LEFT←(~DONE[HEADS])/HEADS
 →(0=⍴LEFT)/FINISH
 READY←{∧/(DONE,1)[(⍵⊃DEP)~⍵]}¨LEFT
 →(~∨/READY)/STUCK
 R←READY/LEFT
 S←SIDE[R]
 M←(S=2)∨S=CUR
 →(∨/M)/GATHER
 CUR←1-CUR
 →TAKE
GATHER:R←M/R
 DONE[R]←1
 ORD←ORD,⍸NODE∊R
 →TAKE
STUCK:EM_BAD←EM_BAD,⊂EM_E,': the schedule found no ready row'
 →0
FINISH:I_F←I_F[ORD]
 I_V←I_V[ORD]
 I_A←I_A[ORD]
 I_B←I_B[ORD]
 I_X←I_X[ORD]
 I_I←I_I[ORD]
 I_R←I_R[ORD]
 S_K←S_K[ORD]
 S_V←S_V[ORD]
 S_S←S_S[ORD]
 S_I←S_I[ORD]
 S_E←S_E[ORD]
∇
