⍝ tk.apl: the tokenizer of the APL compiler. Data-parallel by construction, in
⍝ Hsu's architecture (input/refs/hsu-2019-*.pdf): masks and scans over the whole
⍝ source at once, no branches, no loops, no recursion.
⍝
⍝ TK src: src a character vector with LF line ends. The result is two vectors of
⍝ equal length, one row per token: (TKT TKS), the kind and the text.
⍝ Kinds: 0 LF, 1 name, 2 number, 3 string, 4 glyph. A string keeps its quotes.
⍝
⍝ How each mask works, s being the source:
⍝   a running count reset at each LF: c-⌈\c×s=LF, since a running count of
⍝     non-negative things never falls, ⌈\ of its values at the LFs is its value
⍝     at the latest LF.
⍝   in a string: odd count of quotes so far on the line. In a comment: at or after
⍝     a ⍝ on the line that is outside a string; only characters to its left decide
⍝     that, so quotes inside comments change nothing that came before them.
⍝   a word: a maximal run of letters, digits, ¯ and number dots. A letter first
⍝     makes it a name; anything else first makes it a number: 1E¯6 is one number,
⍝     F16 one name. A dot is a number dot only with a digit after it, so +.× stays
⍝     three glyphs.
⍝   a token starts where the kind changes, and at every LF and glyph.

TK_AL←'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_∆⍙⎕'
TK_DG←'0123456789'
TK_LF←⎕UCS 10

∇R←TK S;LF;QT;Q;STR;CM;COM;WS;WD;KIND;START;KEEP;P
 LF←S=TK_LF
 QT←S=''''
 Q←+\QT
 STR←2|Q-⌈\Q×LF
 CM←+\(S='⍝')∧~STR
 COM←0<CM-⌈\CM×LF
 STR←(STR∨QT)∧~COM
 WS←(S∊' ',⎕UCS 9)∧~STR
 WD←(~STR∨COM)∧(S∊TK_AL,TK_DG,'¯')∨(S='.')∧1↓(S∊TK_DG),0
 ⍝ the kind of each character: 0 LF, 1 word, 3 string, 4 glyph, 5 dropped
 KIND←(4×~LF∨STR∨COM∨WS∨WD)+WD+(3×STR)+5×COM∨WS
 START←(KIND≠¯1↓¯1,KIND)∨KIND∊0 4
 KEEP←KIND≠5
 P←(+\START)×KEEP
 ⍝ a word's kind: name if its first character is a letter, else number
 R←⍸START∧KEEP
 KIND←KIND[R]+(KIND[R]=1)×~S[R]∊TK_AL
 R←KIND(P⊂S)
∇

⍝ The self-check: the tokens, rejoined, are the source without comments and blanks.
∇Z←TKCheck S;R;KEPT
 R←TK S
 KEPT←TKKept S
 Z←KEPT≡∊R[1]
∇

∇Z←TKKept S;LF;QT;Q;STR;CM;COM;WS
 LF←S=TK_LF
 QT←S=''''
 Q←+\QT
 STR←2|Q-⌈\Q×LF
 CM←+\(S='⍝')∧~STR
 COM←0<CM-⌈\CM×LF
 STR←(STR∨QT)∧~COM
 WS←(S∊' ',⎕UCS 9)∧~STR
 Z←(~COM∨WS)/S
∇
