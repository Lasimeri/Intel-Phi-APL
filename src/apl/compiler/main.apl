⍝ main.apl: the compiler's driver: src/apl/qwen35.apl and one model's tensor index (from
⍝ gguf_apl) to one C file over the runtime, the model's compiled code and its tables.
⍝ Runs in GNU APL on the host. Needs SRC_PATH OUT_PATH and the passes (tk ps fl in sh sc em fz).
⍝ A model's shapes come from its tensors, so a compiled file is for one model. A layer is
⍝ compiled once per batch size: NB=1 for a decode step, and the sizes of EM_NBS for a prompt.

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
ROPE_N←ROPE_DIMS÷2
ROPE_COS←2○(⍳CTX)∘.×FREQ_BASE*-(2×⍳ROPE_N)÷ROPE_DIMS
ROPE_SIN←1○(⍳CTX)∘.×FREQ_BASE*-(2×⍳ROPE_N)÷ROPE_DIMS
KV_OF←⌊(⍳HQ)÷HQ÷HKV
⍝ The state's rings, by position: a token tried ahead is taken back by going back. WL: 4 taps and
⍝ 4 positions ahead; SL: a level for each of 5 positions ahead
WL←8
SL←6
⍝ The commonest tokens, the only ones a draft can be: the low ids (ids run by rank)
HOT←32768
NB←1
EM_NBS←16 8 4 3 2
EM_LOGIT_NB←4
SH_C←,¨'D' 'FF' 'HK' 'HV' 'ST' 'HQ' 'HKV' 'HD' 'NK' 'NV' 'CTX' 'RMS_EPS' 'ROPE_N' 'ROPE_COS' 'ROPE_SIN' 'KV_OF' 'NB' 'WL' 'SL' 'HOT'

⍝ ---- the weights: (APL name)(tensor name in the layer). W_CONV is transposed in APL.
MD_ANY←('W_ANORM' 'attn_norm.weight')('W_PNORM' 'post_attention_norm.weight')('W_UP' 'ffn_up.weight')('W_GATE2' 'ffn_gate.weight')('W_DOWN' 'ffn_down.weight')
MD_DELTA←('W_QKV' 'attn_qkv.weight')('W_GATE' 'attn_gate.weight')('W_ALPHA' 'ssm_alpha.weight')('W_BETA' 'ssm_beta.weight')('W_OUT' 'ssm_out.weight')('W_DT' 'ssm_dt.bias')('W_A' 'ssm_a')('W_SNORM' 'ssm_norm.weight')('W_CONV' 'ssm_conv1d.weight')
MD_ATTN←('W_Q' 'attn_q.weight')('W_K' 'attn_k.weight')('W_V' 'attn_v.weight')('W_O' 'attn_output.weight')('W_QN' 'attn_q_norm.weight')('W_KN' 'attn_k_norm.weight')
MD_MTP←('W_EH' 'nextn.eh_proj.weight')('W_ENORM' 'nextn.enorm.weight')('W_HNORM' 'nextn.hnorm.weight')('W_SHN' 'nextn.shared_head_norm.weight')
MD_TOP←('W_EMBD' 'token_embd.weight')('W_ONORM' 'output_norm.weight')('W_OUTPUT' 'output.weight')
SH_ST←,¨'WIN' 'S' 'KC' 'VC'
SH_RI←,¨'p' 'TOK' 'DR' 'AG'

⍝ The tensor index of a bind in layer IL (N_LAYER = the model's own, N_LAYER+1 = the MTP block,
⍝ the file's blk.N_LAYER). A missing output
⍝ matrix is tied to the embedding.
∇Z←IL MDFind R;NAME
 NAME←1⊃R
 →(IL=N_LAYER)/TOP
 NAME←'blk.',(⍕IL⌊N_LAYER),'.',NAME
TOP:Z←TN⍳⊂NAME
 →(Z<⍴TN)/0
 Z←TN⍳⊂'token_embd.weight'
∇

⍝ The APL shape of a bound tensor: a matrix is [ne1 ne0], W_CONV [ne0 ne1].
∇Z←IL MDShape R;T;DM
 T←IL MDFind R
 DM←,T⊃TD
 Z←DM
 →(1=⍴DM)/0
 Z←⌽DM
 →(TT[T]=8)/0
 Z←DM
∇

⍝ ---- the names' shapes, from the first layer of each kind and the model's own
∇MDShapes;FIRST_A;A;B
 FIRST_A←FULL_EVERY-1
 A←MD_ANY,MD_DELTA
 B←MD_ATTN
 SH_W←({0⊃⍵}¨A),({0⊃⍵}¨B),({0⊃⍵}¨MD_TOP),{0⊃⍵}¨MD_MTP
 SH_WS←({0 MDShape ⍵}¨A),({FIRST_A MDShape ⍵}¨B),({N_LAYER MDShape ⍵}¨MD_TOP),{(N_LAYER+1) MDShape ⍵}¨MD_MTP
 SH_Q8←(MDIsQ8¨(0,¨A),(FIRST_A,¨B),(N_LAYER,¨MD_TOP),(N_LAYER+1),¨MD_MTP)/SH_W
 SH_STS←(⊂WL,1↓0 MDShape 8⊃MD_DELTA),(⊂SL,HV,ST,ST),(⊂CTX,HKV,HD),⊂CTX,HKV,HD
∇

∇Z←MDIsQ8 R;T
 T←(0⊃R) MDFind 1↓R
 Z←TT[T]=8
∇

⍝ ---- what each entry takes
∇Z←EMArgs E
 Z←((⊂,'p'),⊂,'X')((⊂⍳0),⊂NB,D)
 →((⊂E)∊'DeltaLayer' 'AttnLayer')/0
 Z←((⊂,'p'),⊂,'X')((⊂⍳0),⊂NB,2×D)
 →(E≡'MtpLayer')/0
 Z←(,⊂,'X')(,⊂NB,D)
 →((⊂E)∊'Hidden' 'Logits' 'DraftLogits')/0
 Z←((⊂'DR'),⊂'AG')((⊂,NB-1),⊂,NB-1)
 →(E≡'Accepted')/0
 Z←(,⊂,'Y')(,⊂,HOT)
 →(E≡'DraftNext')/0
 Z←(,⊂'TOK')(,⊂⍳0)
 →(E≡'EmbedRow')/0
 Z←(,⊂,'⍵')(,⊂,(N_LAYER MDShape 2⊃MD_TOP)[0])
∇

⍝ ---- tables as text. One-dimensional tensors are one row.
∇Z←MDBindLine A;IL;R;T;DM;KIND;RC
 (IL R)←A
 T←IL MDFind R
 DM←,T⊃TD
 RC←⌽2↑DM,1
 KIND←(TT[T]=8)⊃'KIND_F32' 'KIND_Q8'
 KIND←((2=⍴DM)∧TT[T]≠8)⊃KIND 'KIND_F32T'
 Z←' {',(⍕IL),', SLOT_',(0⊃R),', ',(⍕TO[T]),'ull, ',(⍕RC[0]),', ',(⍕RC[1]),', ',KIND,'},'
∇

∇Z←MDLayerBinds IL
 Z←{MDBindLine IL ⍵}¨MD_ANY,(MDIsFull IL)⊃MD_DELTA MD_ATTN
∇

⍝ The MTP block is an attention layer with four more tensors
∇Z←MDMtpBinds
 Z←(MDLayerBinds N_LAYER+1),{MDBindLine (N_LAYER+1) ⍵}¨MD_MTP
∇

∇Z←MDIsFull IL
 Z←(IL>N_LAYER)∨0=FULL_EVERY|IL+1
∇

⍝ The KV cache lives in system memory, in the pinned region where the card's DMA reaches
⍝ it (link.h): one attention layer's KC, then its VC. Its byte offset in the part is a C
⍝ expression; a state with no region place has ¯1.
∇Z←IL MDRegion I;B
 B←4×⌈/,×/¨2↓SH_STS
 Z←'LINK_KV_OFFSET + ',⍕(2×B×⌊IL÷FULL_EVERY)+I×B
∇

∇Z←MDStateLines IL
 →(MDIsFull IL)/ATT
 Z←(⊂' {',(⍕IL),', SLOT_WIN, 1, ',(⍕×/0⊃SH_STS),', -1},'),⊂' {',(⍕IL),', SLOT_S, 1, ',(⍕×/1⊃SH_STS),', -1},'
 →0
ATT:Z←(⊂' {',(⍕IL),', SLOT_KC, 1, ',(⍕×/2⊃SH_STS),', ',(IL MDRegion 0),'},'),⊂' {',(⍕IL),', SLOT_VC, 1, ',(⍕×/3⊃SH_STS),', ',(IL MDRegion 1),'},'
∇

∇Z←T EMTableText V
 Z←'static const float ',T,'[] UNUSED __attribute__((aligned(64))) = {',(¯1↓∊(⍕¨V),¨','),'};'
∇

∇Z←T EMLongText V
 Z←'static const long ',T,'[] UNUSED = {',(¯1↓∊(⍕¨V),¨','),'};'
∇

⍝ ---- a layer function's signature, and a line of the table of batch sizes
∇Z←MDLayerHead N
 Z←'static const float *',N,'(UNUSED struct rt *R, UNUSED float *const *L, UNUSED long p, UNUSED const float *X)'
∇

∇Z←MDTopHead N
 Z←'static const float *',N,'(UNUSED struct rt *R, UNUSED float *const *L, UNUSED const float *X)'
∇

∇Z←MDBatchLine K;S
 S←(K≠1)/⍕K
 Z←' {',(⍕K),', DeltaLayer',S,', AttnLayer',S,', MtpLayer',S,', Hidden',S,', ',((K≤EM_LOGIT_NB)⊃'NULL' ('Logits',S)),', ',(((K≤EM_LOGIT_NB)∧K>1)⊃'NULL' ('Accepted',S)),'},'
∇

⍝ ---- the file
∇MDWrite OUT;FNS;SLOTS;BINDS;STATES;FULLS;HEADTEXT;TAILTEXT;K
 EMReset
 MDShapes
 NB←1
 FNS←(MDLayerHead 'DeltaLayer') EMEntry 'DeltaLayer'
 FNS←FNS,(MDLayerHead 'AttnLayer') EMEntry 'AttnLayer'
 FNS←FNS,(MDLayerHead 'MtpLayer') EMEntry 'MtpLayer'
 FNS←FNS,(MDTopHead 'Hidden') EMEntry 'Hidden'
 FNS←FNS,(MDTopHead 'Logits') EMEntry 'Logits'
 FNS←FNS,(MDTopHead 'DraftLogits') EMEntry 'DraftLogits'
 FNS←FNS,'static long DraftNext(UNUSED struct rt *R, UNUSED float *const *L, UNUSED const float *X)' EMEntry 'DraftNext'
 FNS←FNS,'static const float *EmbedRow(UNUSED struct rt *R, UNUSED float *const *L, UNUSED long TOK)' EMEntry 'EmbedRow'
 FNS←FNS,'static long Next(UNUSED struct rt *R, UNUSED float *const *L, UNUSED const float *X)' EMEntry 'Next'
 K←0
SIZES:→(K≥⍴EM_NBS)/SIZED
 NB←EM_NBS[K]
 FNS←FNS,(MDLayerHead 'DeltaLayer',⍕NB) EMEntry 'DeltaLayer'
 FNS←FNS,(MDLayerHead 'AttnLayer',⍕NB) EMEntry 'AttnLayer'
 FNS←FNS,(MDLayerHead 'MtpLayer',⍕NB) EMEntry 'MtpLayer'
 FNS←FNS,(MDTopHead 'Hidden',⍕NB) EMEntry 'Hidden'
 →(NB>EM_LOGIT_NB)/SIZENEXT
 FNS←FNS,(MDTopHead 'Logits',⍕NB) EMEntry 'Logits'
 FNS←FNS,('static long Accepted',(⍕NB),'(UNUSED struct rt *R, UNUSED float *const *L, UNUSED const long *DR, UNUSED const long *AG)') EMEntry 'Accepted'
SIZENEXT:K←K+1
 →SIZES
SIZED:NB←1
 →(0=⍴EM_BAD)/GOOD
 'COMPILE ERROR: could not lower:'
 ⊣{⎕←⍵}¨∪EM_BAD
 ⎕ES 'compile error'
GOOD:SLOTS←SH_W,SH_ST
 BINDS←↑,/MDLayerBinds¨⍳N_LAYER
 BINDS←BINDS,{MDBindLine N_LAYER ⍵}¨MD_TOP
 BINDS←BINDS,MDMtpBinds
 STATES←↑,/MDStateLines¨(⍳N_LAYER),N_LAYER+1
 FULLS←¯1↓∊(⍕¨0=FULL_EVERY|1+⍳N_LAYER),¨','
 HEADTEXT←(⊂'/* generated by src/apl/compiler from src/apl/qwen35.apl: never edit */')
 HEADTEXT←HEADTEXT,(⊂'#include <stddef.h>'),(⊂'#include <string.h>'),(⊂'#include "rt.h"'),⊂'#include "link.h"'
 HEADTEXT←HEADTEXT,⊂'enum {',(∊{' SLOT_',⍵,','}¨SLOTS),' N_SLOTS };'
 HEADTEXT←HEADTEXT,(EM_TN EMTableText¨EM_TV),EM_LN EMLongText¨EM_LV
 TAILTEXT←⊂'static const unsigned char FULL[] = {',FULLS,'};'
 TAILTEXT←TAILTEXT,(⊂'static const struct tensor_bind TENSORS[] = {'),BINDS,⊂,'};'
 TAILTEXT←TAILTEXT,(⊂'static const struct state_bind STATES[] = {'),STATES,⊂,'};'
 TAILTEXT←TAILTEXT,(⊂'static const struct batch_bind BATCHES[] = {'),(MDBatchLine¨EM_NBS,1),⊂,'};'
 TAILTEXT←TAILTEXT,⊂'const struct model_desc MODEL = {.n_layer = ',(⍕N_LAYER),', .n_slots = N_SLOTS, .split = ',(⍕EM_SPLIT),', .d = ',(⍕D),', .vocab = ',(⍕(N_LAYER MDShape 2⊃MD_TOP)[0]),', .ctx = ',(⍕CTX),', .frame_bytes = ',(⍕EM_FRAME),', .full = FULL, .tensors = TENSORS, .n_tensors = ',(⍕⍴BINDS),', .states = STATES, .n_states = ',(⍕⍴STATES),', .batches = BATCHES, .n_batches = ',(⍕1+⍴EM_NBS),', .embed = EmbedRow, .draft_logits = DraftLogits, .draft_next = DraftNext, .next = Next};'
 (OUT,'.c') MDFile HEADTEXT,FNS,TAILTEXT
 →(~EM_SPLIT)/REPORT
 (OUT,'_host.c') MDFile HEADTEXT,(⊂'#include "vec8.h"'),EM_FH,TAILTEXT
 ⊣{⎕←'placement:' (0⊃⍵) 'card rows' (1⊃⍵) 'host rows' (2⊃⍵) 'side changes' (3⊃⍵)}¨EM_COUNTS
REPORT:⎕←'compiled:' OUT (⍴FNS) 'lines of functions,' (⍴EM_TN) 'float tables,' (⍴EM_LN) 'integer tables, frame' EM_FRAME 'B'
∇

∇P MDFile T;TEXT;FH;I
 TEXT←∊T,¨TK_LF
 TEXT[⍸TEXT='¯']←'-'
 FH←'w' ⎕FIO[3] P
 I←TEXT ⎕FIO[23] FH
 I←⎕FIO[4] FH
∇

PSProgram 19 ⎕CR ⎕UCS ⎕FIO[26] SRC_PATH
FLProgram
EM_SPLIT←0
MDWrite OUT_PATH,'/model'
EM_SPLIT←1
MDWrite OUT_PATH,'/model_card'
)OFF
