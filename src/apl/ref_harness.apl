⍝ ref_harness.apl: run qwen35.apl in GNU APL on real weights: the engine's F64
⍝ TEST REFERENCE, never an inference path. Everything interpreter-only lives
⍝ here: reading and dequantizing the GGUF, binding a layer's weights and state,
⍝ the layer-major driver (one layer loaded once, all tokens through it as one batch,
⍝ with branches and loops), dumps and timing.
⍝
⍝ Needs: the index from gguf_apl, qwen35.apl, and TOKENS OUTDIR LAST_LAYER. Each
⍝ finished layer leaves its output [T D] in OUTDIR/x-LL.txt (⎕PP←17); a rerun
⍝ resumes after the last one. After the last layer: OUTDIR/logits.txt.

⎕IO←0
⎕CT←0
⎕PP←17

⍝ ---- the model constants qwen35.apl declares
D←N_EMBD
FF←N_FF
HK←SSM_GROUP
HV←SSM_RANK
ST←SSM_STATE
HQ←N_HEAD
HKV←N_HEAD_KV
HD←N_HEAD_DIM
NK←HK×ST
NV←HV×ST
CTX←1024
WL←8
SL←6
ROPE_N←ROPE_DIMS÷2
ROPE_COS←2○(⍳CTX)∘.×FREQ_BASE*-(2×⍳ROPE_N)÷ROPE_DIMS
ROPE_SIN←1○(⍳CTX)∘.×FREQ_BASE*-(2×⍳ROPE_N)÷ROPE_DIMS
KV_OF←⌊(⍳HQ)÷HQ÷HKV


⍝ ---- bind a layer's weights and zero its state. (The matrices are held only
⍝ while the layer runs: a large one is 600 MB in GNU APL.)
∇IL Bind FULL
 W_ANORM←Weight IL L 'attn_norm.weight'
 W_PNORM←Weight IL L 'post_attention_norm.weight'
 W_UP←Weight IL L 'ffn_up.weight'
 W_GATE2←Weight IL L 'ffn_gate.weight'
 W_DOWN←Weight IL L 'ffn_down.weight'
 →FULL/ATT
 W_QKV←Weight IL L 'attn_qkv.weight'
 W_GATE←Weight IL L 'attn_gate.weight'
 W_ALPHA←Weight IL L 'ssm_alpha.weight'
 W_BETA←Weight IL L 'ssm_beta.weight'
 W_OUT←Weight IL L 'ssm_out.weight'
 W_DT←Weight IL L 'ssm_dt.bias'
 W_A←Weight IL L 'ssm_a'
 W_SNORM←Weight IL L 'ssm_norm.weight'
 W_CONV←⍉Weight IL L 'ssm_conv1d.weight'
 WIN←(WL,1↓⍴W_CONV)⍴0
 S←(SL,HV,ST,ST)⍴0
 →0
ATT:W_Q←Weight IL L 'attn_q.weight'
 W_K←Weight IL L 'attn_k.weight'
 W_V←Weight IL L 'attn_v.weight'
 W_O←Weight IL L 'attn_output.weight'
 W_QN←Weight IL L 'attn_q_norm.weight'
 W_KN←Weight IL L 'attn_k_norm.weight'
 KC←(((⍴TOKENS),HKV,HD)⍴0)
 VC←KC
∇

⍝ Free a layer's matrices before the next one is read
∇Unbind;N
 N←⎕EX¨'W_ANORM' 'W_PNORM' 'W_UP' 'W_GATE2' 'W_DOWN' 'W_QKV' 'W_GATE' 'W_ALPHA' 'W_BETA' 'W_OUT' 'W_DT' 'W_A' 'W_SNORM' 'W_CONV' 'WIN' 'S' 'W_Q' 'W_K' 'W_V' 'W_O' 'W_QN' 'W_KN' 'KC' 'VC'
∇

⍝ ---- the embedding of token ids [T D]: one row seek per token
∇Z←Embed W;TE;I
 TE←Find 'token_embd.weight'
 Z←((⍴W),D)⍴0
 I←0
NEXT:→(I≥⍴W)/0
 Z[I;]←,TE Q8Rows W[I],1
 I←I+1
 →NEXT
∇

⍝ ---- the logits of one hidden vector, the output matrix in row chunks, never
⍝ whole: it is 248320 x D. Same statement as Logits in qwen35.apl, over chunks.
∇Z←ChunkedLogits X;T;N;R;C;V
 W_ONORM←Weight 'output_norm.weight'
 V←W_ONORM RMS X
 T←TN⍳⊂'output.weight'
 T←(T=⍴TN)⊃T,Find 'token_embd.weight'
 N←(⊃TD[T])[1]
 Z←N⍴0
 R←0
NEXT:→(R≥N)/0
 C←8192⌊N-R
 Z[R+⍳C]←(T Q8Rows R,C)+.×V
 R←R+C
 →NEXT
∇

⍝ ---- files and time
∇Z←Clock
 Z←24 60 60 1000⊥¯4↑⎕TS
∇

∇Z←Path IL
 Z←OUTDIR,'/x-',(¯2↑'0',⍕IL),'.txt'
∇

∇N WriteText P;F;R
 F←'w' ⎕FIO[3] P
 R←(⍕,N) ⎕FIO[23] F
 R←⎕FIO[4] F
∇

∇Z←ReadNumbers P
 Z←⍎19 ⎕CR ⎕UCS ⎕FIO[26] P
∇

∇Z←Exists P;F
 F←⎕FIO[3] P
 Z←F≥0
 →(~Z)/0
 F←⎕FIO[4] F
∇

⍝ The last finished layer below N_LAYER, or ¯1
∇Z←LastDone;I
 Z←¯1
 I←0
NEXT:→((I≥N_LAYER)∨~Exists Path I)/0
 Z←I
 I←I+1
 →NEXT
∇

⍝ ---- one layer over all tokens: bind once, the whole prompt as one batch from position 0.
⍝ NB is read by qwen35.apl through dynamic scope, like the rest of the model's constants.
∇Z←IL RunLayer X;FULL;NB
 FULL←0=FULL_EVERY|IL+1
 IL Bind FULL
 NB←↑⍴X
 →FULL/A
 Z←0 DeltaLayer X
 →DONE
A:Z←0 AttnLayer X
DONE:Unbind
∇

∇Main;T;IL;X;T0;STOP;LG;BEST
 T←⍴TOKENS
 IL←LastDone
 STOP←(LAST_LAYER<0)⊃LAST_LAYER,N_LAYER-1
 →(IL<0)/FRESH
 X←(T,D)⍴ReadNumbers Path IL
 'resuming after layer' IL
 IL←IL+1
 →LOOP
FRESH:T0←Clock
 X←Embed TOKENS
 'embedded' T 'tokens in ms' ((Clock)-T0)
 IL←0
LOOP:→(IL>STOP)/AFTER
 T0←Clock
 X←IL RunLayer X
 ((⍕IL),' ',((0=FULL_EVERY|IL+1)⊃'delta' 'full '),' ms',⍕(Clock)-T0),' maxabs ',⍕⌈/,|X
 X WriteText Path IL
 IL←IL+1
 →LOOP
AFTER:→(STOP<N_LAYER-1)/0
 T0←Clock
 LG←ChunkedLogits X[↑T-1;]
 LG WriteText OUTDIR,'/logits.txt'
 BEST←Next LG
 'logits ms' ((Clock)-T0) 'top token' BEST 'value' (LG[BEST])
∇

FH←⎕FIO[3] GGUF_PATH
Main
⊣⎕FIO[4] FH
)OFF
