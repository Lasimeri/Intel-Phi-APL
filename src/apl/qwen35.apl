⍝ qwen35.apl: one layer of the GGUF "qwen35" architecture (Qwen3.5-family hybrid: Gated
⍝ DeltaNet layers, a gated full-attention layer every FULL_EVERY-th) over NB tokens at once:
⍝ X [NB D] holds the tokens at positions p+⍳NB, and a decode step is simply NB=1. This file is
⍝ the engine's only statement of the model's math. The APL
⍝ compiler (src/apl/compiler/) lowers it to the KNC VPU; GNU APL runs the same
⍝ text in F64 as the test reference (src/apl/ref_harness.apl).
⍝
⍝ The compilable unit: straight-line statements, no branches, no loops, no file
⍝ access. Dialect: what GNU APL 2.0 executes, ⎕IO←0, ⎕CT←0, ⎕PP←17.
⍝
⍝ Names this file does not define, all declared here:
⍝   MODEL CONSTANTS (fixed per model; the compiler folds them):
⍝     D FF HK HV ST HQ HKV HD NK NV CTX RMS_EPS ROPE_N ROPE_COS ROPE_SIN KV_OF NB WL SL HOT
⍝   WEIGHTS of the current layer (read-only):
⍝     W_ANORM W_PNORM W_UP W_GATE2 W_DOWN        every layer
⍝     W_QKV W_GATE W_ALPHA W_BETA W_DT W_A W_CONV W_SNORM W_OUT   DeltaNet
⍝     W_Q W_K W_V W_O W_QN W_KN                  attention
⍝     W_EH W_ENORM W_HNORM W_SHN                 the MTP block, with W_Q W_K...
⍝   and of the model: W_ONORM W_OUTPUT
⍝   STATE of the current layer (updated in place):
⍝     WIN [WL C] the conv ring; S [SL HV ST ST] with S[l;h;j;i] = state(i,j) of head h
⍝     KC VC [CTX HKV HD] the KV cache
⍝   All three are indexed by position, level l = SL|p being the state before position p. To
⍝   take back tokens tried ahead (speculative decoding), start again from an earlier position:
⍝   nothing to copy, as long as no more than WL-4 and SL-1 positions lie beyond it.
⍝
⍝ Shapes in comments: [rows cols]. A weight matrix is [out in], as GGUF stores it
⍝ (ne0 = in, contiguous), so a projection is W+.×x. p is the token's position.
⍝
⍝ Math: llama.cpp f5b9bd3 src/models/qwen35.cpp, delta-net-base.cpp, ggml-cpu
⍝ ops.cpp (gated_delta_net, ssm_conv, rope, l2_norm). Held against it by
⍝ tests/host/logit_cmp (the 2B: top-1 same, NRMSE 0.0215, top 10 identical).

⍝ ---- scalar functions; the compiler fuses them into the surrounding pass. The
⍝ exponent is capped at 700: in GNU APL 2×∞ is a DOMAIN ERROR, e*700 is finite in
⍝ F64, and these functions are flat long before it.
Sig←{÷1+*700⌊-⍵}
SiLU←{⍵×Sig ⍵}
SoftPlus←{⍟1+*700⌊⍵}

⍝ ---- softmax of a vector. A function with a declared local, not a lambda: in GNU APL
⍝ a name assigned in a lambda is global, one array shared across a parallel map.
∇Z←SoftMax X;E
 E←*X-⌈/X
 Z←E÷+/E
∇

⍝ ---- norms of a vector: W RMS x, and x ÷ max(‖x‖, eps)
RMS←{⍺×⍵×(RMS_EPS+(+/⍵×⍵)÷≢⍵)*¯0.5}
L2←{⍵÷RMS_EPS⌈(+/⍵×⍵)*0.5}

⍝ ---- rotary position of one head's vector at position p: IMROPE for text, all
⍝ sections take position p (checked: sections 11 11 10 0 cover all 32 pairs); the
⍝ pairs are (i, i+ROPE_N), the rest passes through. The tables are compile-time
⍝ constants: ROPE_COS[p;i] is 2○p×FREQ_BASE*-2i÷2×ROPE_N.
Rope←{(((ROPE_N↑⍵)×ROPE_COS[⍺;])-(ROPE_N↑ROPE_N↓⍵)×ROPE_SIN[⍺;]),(((ROPE_N↑⍵)×ROPE_SIN[⍺;])+(ROPE_N↑ROPE_N↓⍵)×ROPE_COS[⍺;]),(2×ROPE_N)↓⍵}

⍝ ---- the gated delta rule of one value head, from its part of the state at level SO to its
⍝ part at SN, reading key head h mod HK. E B V K Q SO SN are DeltaMix's. Heads share nothing, and a head's part stays in
⍝ cache through all its passes. (E×S)+.×K is E×S+.×K: the old state is only read.
∇Z←DeltaHead h;A;C;N
 A←S[SO;h;;]+.×K[HK|h;]
 C←S[SO;h;;]+.×Q[HK|h;]
 N←B[h]×V[h;]-E[h]×A
 S[SN;h;;]←(E[h]×S[SO;h;;])+N∘.×K[HK|h;]
 Z←(E[h]×C)+N×+/K[HK|h;]×Q[HK|h;]
∇

⍝ ---- the gated DeltaNet mixer of one token, row t of the batch, at position p+t. The
⍝ projections are DeltaStep's locals: this reads what they made of its token.
∇Z←DeltaMix t;P;SO;SN;GT;B;G;U;Q;K;V;E;O
 P←p+t
 SO←SL|P
 SN←SL|P+1
 WIN[WL|P;]←QKV[t;]
 GT←GATE[t;]
 B←Sig BETA[t;]
 G←W_A×SoftPlus W_DT+ALPHA[t;]
 ⍝ depthwise causal conv: 4 taps over the ring, oldest first; then q | k | v
 U←SiLU+⌿WIN[WL|P+WL-3 2 1 0;]×W_CONV
 Q←L2⍤1⊢HK ST⍴NK↑U
 K←L2⍤1⊢HK ST⍴NK↑NK↓U
 V←HV ST⍴(2×NK)↓U
 E←*G
 O←(÷ST*0.5)×DeltaHead⍤0⊢⍳HV
 ⍝ the gated norm: what the output projection takes
 Z←,(W_SNORM RMS⍤1⊢O)×SiLU HV ST⍴GT
∇

⍝ ---- the mixer over the batch: H [NB D] the normed input. One pass over each matrix serves
⍝ every token; the tokens then take their turns on the state, in order.
∇Z←p DeltaStep H;QKV;GATE;BETA;ALPHA
 QKV←H+.×⍉W_QKV
 GATE←H+.×⍉W_GATE
 BETA←H+.×⍉W_BETA
 ALPHA←H+.×⍉W_ALPHA
 Z←(DeltaMix⍤0⊢⍳NB)+.×⍉W_OUT
∇

⍝ ---- the gated full-attention mixer of one token. Each query head reads its kv head's
⍝ history (GQA): KV_OF selects them, then one inner product per head over positions 0..P,
⍝ the same form as the delta rule's.
∇Z←AttnMix t;P;QF;Q;KQ;VQ;A
 P←p+t
 QF←HQ 2 HD⍴PQ[t;]
 Q←P Rope⍤0 1⊢W_QN RMS⍤1⊢QF[;0;]
 KC[P;;]←P Rope⍤0 1⊢W_KN RMS⍤1⊢HKV HD⍴PK[t;]
 VC[P;;]←HKV HD⍴PV[t;]
 KQ←1 0 2⍉KC[⍳P+1;KV_OF;]
 VQ←1 0 2⍉VC[⍳P+1;KV_OF;]
 A←SoftMax⍤1⊢(÷HD*0.5)×KQ(+.×)⍤2 1⊢Q
 Z←A(+.×)⍤1 2⊢VQ
 Z←(,Z)×Sig,QF[;1;]
∇

∇Z←p AttnStep H;PQ;PK;PV
 PQ←H+.×⍉W_Q
 PK←H+.×⍉W_K
 PV←H+.×⍉W_V
 Z←(AttnMix⍤0⊢⍳NB)+.×⍉W_O
∇

⍝ ---- the SwiGLU feed-forward of the batch
∇Z←FFN H
 Z←((SiLU H+.×⍉W_GATE2)×H+.×⍉W_UP)+.×⍉W_DOWN
∇

⍝ ---- one layer: pre-norm, mixer, residual; post-norm, FFN, residual. The two
⍝ kinds are separate functions; which one a layer is comes from its index
⍝ ((il+1) mod FULL_EVERY = 0 is full attention), fixed at compile time.
∇Z←p DeltaLayer X
 Z←X+p DeltaStep W_ANORM RMS⍤1⊢X
 Z←Z+FFN W_PNORM RMS⍤1⊢Z
∇

∇Z←p AttnLayer X
 Z←X+p AttnStep W_ANORM RMS⍤1⊢X
 Z←Z+FFN W_PNORM RMS⍤1⊢Z
∇

⍝ ---- the MTP block, a layer that drafts the next token: X [NB 2×D], each row a token's
⍝ embedding, then the hidden state of the position before it
∇Z←MtpRow A
 Z←(W_ENORM RMS D↑A),W_HNORM RMS D↓A
∇

∇Z←p MtpLayer X
 Z←p AttnLayer (MtpRow⍤1⊢X)+.×⍉W_EH
 Z←W_SHN RMS⍤1⊢Z
∇

⍝ ---- after the last layer: the hidden states [NB D], the logits [NB vocab] of hidden states,
⍝ and greedy selection of a row (the lowest id on ties).
∇Z←Hidden X
 Z←W_ONORM RMS⍤1⊢X
∇

∇Z←Logits X
 Z←X+.×⍉W_OUTPUT
∇
Next←{⍵⍳⌈/⍵}

⍝ ---- a token's embedding: one row of the embedding matrix
∇Z←EmbedRow TOK
 Z←W_EMBD[TOK;]
∇

⍝ ---- a draft's logits, only over the HOT commonest tokens (the ids run by rank): a draft is only
⍝ a guess, the pass that checks it reads the whole output matrix
∇Z←DraftLogits X
 Z←X+.×⍉HOT D↑W_OUTPUT
∇

∇Z←DraftNext Y
 Z←Next Y
∇

⍝ ---- how many of the drafts DR stand: the leading ones the model agrees with, AG
∇Z←DR Accepted AG
 Z←+/∧\DR=AG
∇
