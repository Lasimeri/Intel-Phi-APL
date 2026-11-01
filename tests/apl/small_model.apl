⍝ small_model.apl: qwen35.apl's functions run on a small synthetic model (TEST
⍝ ONLY): every shape small, weights from a fixed generator. Prints the outputs of
⍝ three tokens of a DeltaNet layer, three of an attention layer, and the logits
⍝ at ⎕PP←17, so two versions of the source can be compared by their text.
⎕IO←0
⎕CT←0
⎕PP←17
⎕PW←1000
D←32
NB←1
WL←8
HOT←16
SL←6
FF←48
HK←2
HV←4
ST←8
HQ←4
HKV←2
HD←8
NK←HK×ST
NV←HV×ST
CTX←8
RMS_EPS←1E¯6
ROPE_N←2
ROPE_COS←2○(⍳CTX)∘.×10000000*-(2×⍳ROPE_N)÷2×ROPE_N
ROPE_SIN←1○(⍳CTX)∘.×10000000*-(2×⍳ROPE_N)÷2×ROPE_N
KV_OF←⌊(⍳HQ)÷HQ÷HKV
SEED←12345
∇Z←Rnd S;N
 N←×/S
 Z←S⍴(¯1+2×(1000003|SEED+7919×⍳N)÷1000003)
 SEED←1000003|SEED+7919×N
∇
W_ANORM←Rnd ,D
W_PNORM←Rnd ,D
W_UP←Rnd FF,D
W_GATE2←Rnd FF,D
W_DOWN←Rnd D,FF
W_QKV←Rnd (NK+NK+NV),D
W_GATE←Rnd NV,D
W_ALPHA←Rnd HV,D
W_BETA←Rnd HV,D
W_DT←Rnd ,HV
W_A←-|Rnd ,HV
W_SNORM←Rnd ,ST
W_CONV←Rnd 4,NK+NK+NV
W_OUT←Rnd D,NV
WIN←(WL,NK+NK+NV)⍴0
S←SL,HV,ST,ST
S←S⍴0
W_Q←Rnd (HQ×2×HD),D
W_K←Rnd (HKV×HD),D
W_V←Rnd (HKV×HD),D
W_O←Rnd D,HQ×HD
W_QN←Rnd ,HD
W_KN←Rnd ,HD
KC←CTX,HKV,HD
KC←KC⍴0
VC←KC
W_EH←Rnd D,2×D
W_ENORM←Rnd ,D
W_HNORM←Rnd ,D
W_SHN←Rnd ,D
W_ONORM←Rnd ,D
W_OUTPUT←Rnd 64,D
X←Rnd 3,D
⍕,(0 DeltaLayer 1 D⍴X[0;]),(1 DeltaLayer 1 D⍴X[1;]),2 DeltaLayer 1 D⍴X[2;]
⍕,(0 AttnLayer 1 D⍴X[0;]),(1 AttnLayer 1 D⍴X[1;]),2 AttnLayer 1 D⍴X[2;]
⍕,3 MtpLayer 1 (2×D)⍴X[0;],X[1;]
⍕,Hidden 1 D⍴X[2;]
⍕,Logits Hidden 1 D⍴X[2;]
⍕Next ,Logits Hidden 1 D⍴X[2;]
⍕,DraftLogits Hidden 1 D⍴X[2;]
⍕DraftNext ,DraftLogits Hidden 1 D⍴X[2;]
⍕1 1 0 Accepted 1 1 1
⍕,S
⍕,WIN
⍕,KC
)OFF
