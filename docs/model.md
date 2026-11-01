# The model in APL

`src/apl/qwen35.apl` is the engine's only statement of the model's math: the GGUF `qwen35`
architecture, a hybrid of Gated DeltaNet layers with a gated full-attention layer every
`FULL_EVERY`-th. This page reads it statement by statement. Dialect: what GNU APL 2.0 executes,
`⎕IO←0`, `⎕CT←0`. The compilable unit is straight-line statements: no branches, no loops, no file access.

## Constants

| Name | Meaning | 2B | 9B |
|---|---|---|---|
| `D` | model width | 2048 | 4096 |
| `FF` | FFN hidden width | 6144 | 12288 |
| layers | total (delta + attention) | 24 (18 + 6) | 32 (24 + 8) |
| `HV` | value heads | 16 | 32 |
| `HK` | key heads | 16 | 16 |
| `ST` | state width | 128 | 128 |
| `HQ` | attention query heads | 8 | 16 |
| `HKV` | attention key/value heads | 2 | 4 |
| `HD` | attention head width | 256 | 256 |

Derived: `NK←HK×ST`, `NV←HV×ST`, conv width `C←NK+NK+NV`. `NB` is the tokens in a batch, `WL` (8) and
`SL` (6) the ring and state levels, `HOT` (32,768) the tokens a draft can be.

Scalar functions; the `700⌊` exists only because `2×∞` is a DOMAIN ERROR in GNU APL's F64, and the
compiled F32 `exp` already clamps, so the compiler removes it:

```apl
Sig←{÷1+*700⌊-⍵}            ⍝ logistic sigmoid
SiLU←{⍵×Sig ⍵}
SoftPlus←{⍟1+*700⌊⍵}
RMS←{⍺×⍵×(RMS_EPS+(+/⍵×⍵)÷≢⍵)*¯0.5}
L2←{⍵÷RMS_EPS⌈(+/⍵×⍵)*0.5}
```

## A delta layer

```apl
∇Z←p DeltaLayer X
 Z←X+p DeltaStep W_ANORM RMS⍤1⊢X
 Z←Z+FFN W_PNORM RMS⍤1⊢Z
∇
```

Pre-norm, mixer, residual; post-norm, FFN, residual. `X` is `[NB D]`, tokens at positions `p+⍳NB`. A
decode step is simply `NB=1`.

`DeltaStep` makes four projections with one pass over each matrix serving every token, then the tokens
take their turns on the state:

```apl
QKV←H+.×⍉W_QKV
GATE←H+.×⍉W_GATE
BETA←H+.×⍉W_BETA
ALPHA←H+.×⍉W_ALPHA
Z←(DeltaMix⍤0⊢⍳NB)+.×⍉W_OUT
```

`DeltaMix` handles one token, row `t`, at position `P←p+t`:

| Statement | Meaning |
|---|---|
| `WIN[WL\|P;]←QKV[t;]` | write the projection into the conv ring |
| `B←Sig BETA[t;]` | the update's gate, per head |
| `G←W_A×SoftPlus W_DT+ALPHA[t;]`, `E←*G` | the state's decay, per head |
| `U←SiLU+⌿WIN[WL\|P+WL-3 2 1 0;]×W_CONV` | depthwise causal conv: 4 taps, oldest first |
| `Q←L2⍤1⊢HK ST⍴NK↑U`, `K←L2⍤1⊢HK ST⍴NK↑NK↓U` | query and key, normalized per key head |
| `V←HV ST⍴(2×NK)↓U` | value, per value head |
| `O←(÷ST*0.5)×DeltaHead⍤0⊢⍳HV` | the delta rule, each head |
| `Z←,(W_SNORM RMS⍤1⊢O)×SiLU HV ST⍴GT` | the gated norm |

## The delta rule

Head `h` holds a state `S`, `[ST ST]`, at 6 levels indexed by position. Level `SL|P` is the state before
position `P`; `S_new` goes to `SL|P+1`, out of place. Written as first stated:

```
N      = B x (V - E x (S_old . K))
S_new  = E x S_old + N o K
Z      = S_new . Q
```

The engine computes the same thing without reading `S_new` within the token:

```apl
∇Z←DeltaHead h;A;C;N
 A←S[SO;h;;]+.×K[HK|h;]
 C←S[SO;h;;]+.×Q[HK|h;]
 N←B[h]×V[h;]-E[h]×A
 S[SN;h;;]←(E[h]×S[SO;h;;])+N∘.×K[HK|h;]
 Z←(E[h]×C)+N×+/K[HK|h;]×Q[HK|h;]
∇
```

This follows from `S_new . Q = E x (S_old . Q) + N x (K . Q)`. `tests/apl/delta_witness.sh` holds the two
forms against each other. Heads share nothing, so the map over `h` runs on the host's threads.

## An attention layer

```apl
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
```

Each query head reads its key/value head's history (`KV_OF` selects them, GQA), over positions 0 to `P`.
`Rope` is IMROPE with compile-time tables, all sections taking position `P`. The history arrays `KQ`, `VQ`
are never built: the compiler has the inner products read the KV cache where it lies.

## The FFN

```apl
∇Z←FFN H
 Z←((SiLU H+.×⍉W_GATE2)×H+.×⍉W_UP)+.×⍉W_DOWN
∇
```

SwiGLU. On the 9B its three matrices are 61 % of what a token reads. A decode token reads 8.43 GB: FFN 5.13,
delta matrices 1.72, output head 1.08, attention 0.50.

## Drafting ahead

Both models carry a draft head as tensors `blk.N.nextn.*`, where N is the layer count:

```apl
∇Z←MtpRow A
 Z←(W_ENORM RMS D↑A),W_HNORM RMS D↓A
∇
∇Z←p MtpLayer X
 Z←p AttnLayer (MtpRow⍤1⊢X)+.×⍉W_EH
 Z←W_SHN RMS⍤1⊢Z
∇
∇Z←DraftLogits X
 Z←X+.×⍉HOT D↑W_OUTPUT
∇
∇Z←DR Accepted AG
 Z←+/∧\DR=AG
∇
```

`X` here is `[NB 2×D]`: a token's embedding, then the hidden state of the position before it. Token ids run
by rank, so `HOT D↑W_OUTPUT` is the commonest tokens' rows.

## After the last layer

```apl
∇Z←Hidden X
 Z←W_ONORM RMS⍤1⊢X
∇
∇Z←Logits X
 Z←X+.×⍉W_OUTPUT
∇
Next←{⍵⍳⌈/⍵}
```

Greedy selection, the lowest id on ties. Sampling is not built.

## Held against

llama.cpp f5b9bd3, `src/models/qwen35.cpp`, `delta-net-base.cpp`, `ggml-cpu/ops.cpp`. On the 2B the F64
reference has the same top-1, NRMSE 0.0215 and top 10 identical; see [testing.md](testing.md).
