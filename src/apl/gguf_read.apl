⍝ gguf_read.apl: read and dequantize a GGUF model's tensors in APL, given the
⍝ index from gguf_apl and an open handle FH on the file. C does the indexing;
⍝ APL does the arithmetic (checked against C: scripts/apl_ref.sh dq).

⎕IO←0
⎕CT←0
⎕PP←17

⍝ ---- reading the model: APL dequantizes the GGUF itself. IEEE bit patterns to
⍝ values, exact in F64 (no inf or NaN expected in the block scales).
∇Z←F16 U;E;M
 E←⌊(32768|U)÷1024
 M←1024|U
 Z←(1-2×U≥32768)×((E=0)×M×2*¯24)+(E≠0)×(1+M÷1024)×2*E-15
∇

∇Z←F32 U;E;M
 E←⌊((2*31)|U)÷2*23
 M←(2*23)|U
 Z←(1-2×U≥2*31)×((E=0)×M×2*¯149)+(E≠0)×(1+M÷2*23)×2*E-127
∇

⍝ A[1] bytes at file offset A[0], as integers 0..255
∇Z←Bytes A;R
 R←A[0] ⎕FIO[13] FH
 Z←A[1] ⎕FIO[6] FH
 →(A[1]=⍴Z)/0
 ⎕ES 'short read from the model file'
∇

∇Z←Find N
 Z←TN⍳⊂N
 →(Z<⍴TN)/0
 ⎕ES 'no tensor named ',N
∇

⍝ T Q8Rows first count: rows of Q8_0 tensor T as [count ne0], block scale × int8
⍝ (exact in F64). A block is a 2-byte f16 scale and 32 int8 quants.
∇Z←T Q8Rows A;DIM;NB;B;SC
 DIM←(⊃TD[T])[0]
 NB←DIM÷32
 B←(A[1],NB,34)⍴Bytes (TO[T]+A[0]×NB×34),A[1]×NB×34
 SC←F16 B[;;0]+256×B[;;1]
 B←B[;;2+⍳32]
 Z←(A[1],DIM)⍴SC×[0 1]B-256×B≥128
∇

⍝ A whole tensor by name: Q8_0 (ggml type 8) or F32 (0), row-major [.. ne1 ne0]
∇Z←Weight N;T;R;B
 T←Find N
 →(TT[T]=8)/Q8
 R←⌽⊃TD[T]
 B←((×/R),4)⍴Bytes TO[T],4××/R
 Z←R⍴F32 B[;0]+256×B[;1]+256×B[;2]+256×B[;3]
 →0
Q8:Z←T Q8Rows 0,(⊃TD[T])[1]
∇

⍝ The name of a tensor of layer IL
∇Z←IL L N
 Z←'blk.',(⍕IL),'.',N
∇

