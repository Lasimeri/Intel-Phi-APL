⍝ cell.apl: the cell of a map region for the inliner's oracle (TEST ONLY). R CELL_ I S:
⍝ S itself when its rank is at most R, else its item I.
∇Z←R CELL_ A;I;S
 (I S)←A
 Z←S
 →(R≥⍴⍴S)/0
 →(1≠⍴⍴S)/ROW
 Z←S[I]
 →0
ROW:Z←S[I;]
∇
