/* rt.h: the runtime under the compiled APL: the primitives generated code calls, and
 * the description of a compiled model the engine's driver reads. The math of the
 * model is in src/apl/qwen35.apl; these are the machine forms of APL's primitives.
 *
 * Array arguments are 64-byte aligned and hold whole 16-lane blocks (a buffer's tail
 * lanes may be written). rt_ew, rt_sum and rt_maxv take any length; the inner products
 * and rank kernels need a last axis that is a multiple of 16. Large kernels run on the
 * team, others on the calling thread: a team round costs about 20 us, 25,000 cycles. */
#ifndef APL_RT_H
#define APL_RT_H
#include <stddef.h>
#include <stdint.h>

#define UNUSED __attribute__((unused))

struct team;
struct gemm_chain;

struct rt {
    struct team *team;
    char *frame; /* the temporaries of the running entry */
    void *chain; /* the card: struct gemv_chain, RT_BATCH per thread (rt_init) */
    struct gemm_chain *gemm_chain; /* the card: one per thread (rt_init) */
    int threads;
    int last; /* the host: this token is the last of its batch (the token map sets it) */
};

/* ---- elementwise: z[i] = a[i] op b[i]; V an array, S the scalar s */
enum {
    RT_ADD_VV, RT_ADD_VS, RT_ADD_SV,
    RT_SUB_VV, RT_SUB_VS, RT_SUB_SV,
    RT_MUL_VV, RT_MUL_VS, RT_MUL_SV,
    RT_DIV_VV, RT_DIV_VS, RT_DIV_SV,
    RT_MAX_VV, RT_MAX_VS, RT_MAX_SV,
    RT_MIN_VV, RT_MIN_VS, RT_MIN_SV,
    RT_NEG, RT_RCP, RT_EXP, RT_LOG /* monadic -, ÷, *, ⍟ of b */
};
/* The card: once, before any product. 0, or -1 without memory. */
int rt_init(struct rt *R, int threads);

void rt_ew(struct rt *R, int op, float *z, const float *a, const float *b, float s, long n);

/* ---- scalars, in the vector unit */
float rt_s_add(float a, float b);
float rt_s_sub(float a, float b);
float rt_s_mul(float a, float b);
float rt_s_div(float a, float b);
float rt_s_max(float a, float b);
float rt_s_min(float a, float b);
float rt_s_rsqrt(float x); /* x*¯0.5 */
float rt_s_sqrt(float x);  /* x*0.5 */

/* ---- reductions: +/a, ⌈/a; a⍳v (bit-identical match, the lowest index) */
float rt_sum(const float *a, long n);
float rt_maxv(const float *a, long n);
long rt_index_of(const float *a, long n, float v);

/* +⌿a for a [rows cols] */
void rt_redfirst(float *z, const float *a, long rows, long cols);

/* ---- products */
void rt_gemv(struct rt *R, const void *w, long rows, long cols, const float *x, float *y); /* Q8_0, form I */
/* Adjacent products: queue them, then one team round runs them all, a thread doing its share
 * of each in turn with no barrier between. At most RT_BATCH, none reading another's result. */
#define RT_BATCH 8
void rt_gemv_q(struct rt *R, const void *w, long rows, long cols, const float *x, float *y);
/* Before the wait for x, the queue complete: the round opens, the threads read ahead; go runs it */
void rt_gemv_open(struct rt *R);
void rt_gemv_go(struct rt *R);
/* x+.×⍉w, x [t cols] and y [t rows]: one pass over the weights serves t tokens (gemm.h) */
void rt_gemm(struct rt *R, const void *w, long rows, long cols, const float *x, long t, float *y);
void rt_gemm_q(struct rt *R, const void *w, long rows, long cols, const float *x, long t, float *y); /* in the round of rt_gemv_go */
void rt_ip21(struct rt *R, float *z, const float *a, long h, long r, long c, const float *b); /* a(+.×)⍤2 1⊢b */
void rt_ip12(struct rt *R, float *z, const float *a, long h, long t, long c, const float *b); /* a(+.×)⍤1 2⊢b */
void rt_outer11(struct rt *R, float *z, const float *a, const float *b, long h, long m, long k); /* a(∘.×)⍤1 1⊢b */
void rt_outer_add(struct rt *R, float *z, const float *a, const float *b, long h, long m, long k); /* z←z+a(∘.×)⍤1 1⊢b, fused */
void rt_scale_outer(struct rt *R, float *z, const float *s, float g, const float *a, const float *b, long m, long k); /* z←(g×s)+a∘.×b, fused */
void rt_scale0n(struct rt *R, float *z, const float *a, const float *b, long h, long cell); /* a×⍤0 k⊢b */
/* The same inner products over a head's history where it lies: for a state kv [C n1 c],
 * b's h-th item is kv[;l[h];] for positions 0..t-1. Nothing is gathered or transposed. */
void rt_ip21_kv(struct rt *R, float *z, const float *kv, long n1, long c, const long *l, long h, long t, const float *b);
void rt_ip12_kv(struct rt *R, float *z, const float *a, const float *kv, long n1, long c, const long *l, long h, long t);

/* ---- structure */
void rt_q8_row(const void *w, long cols, long row, float *z); /* W[row;], W Q8_0 in form I */
void rt_cat(float *z, const float *a, long na, const float *b, long nb);
/* z←a[l0;l1;] for a [n0 n1 n2]: k0, k1 items; a NULL list is ⍳k */
void rt_index3(struct rt *R, float *z, const float *a, long n1, long n2, const long *l0, long k0, const long *l1, long k1);
void rt_transpose102(struct rt *R, float *z, const float *a, long n0, long n1, long n2); /* 1 0 2⍉a */

/* ---- the host's worker only (src/host/rt_host.c) */
void rt_host_set_par(long lanes);
void rt_host_bind(struct rt *R, int threads);
void rt_fz(struct rt *R, void (*fn)(void *arg, int tid, int n), void *arg, long lanes);
/* A region of a map the compiler outlined: on every thread of the team, or this one */
void rt_par(struct rt *R, void (*fn)(void *arg, int tid, int n), void *arg);

/* ---- a compiled model. Only pointers and integers cross to the engine's driver. */
enum { KIND_F32, KIND_Q8, KIND_F32T }; /* F32T: [cols rows] in APL, [rows cols] in the file */

struct tensor_bind {
    int layer; /* n_layer = the model's own */
    int slot;
    uint64_t offset;
    long rows, cols;
    int kind;
};

struct state_bind {
    int layer, slot, side; /* side: 0 the card, 1 the host (a split model only) */
    long elems;
    long region_off; /* >= 0: this state is in system memory, at this offset of the region */
};

typedef const float *(*layer_fn)(struct rt *R, float *const *L, long p, const float *X);

typedef const float *(*top_fn)(struct rt *R, float *const *L, const float *X);

/* A model's parts compiled for nb tokens at once, X [nb d] (mtp [nb 2d]): the tokens at positions
 * p .. p+nb-1. hidden: the normed last output; logits, of those (NULL for large nb); mtp: the
 * block that drafts, its slots the layer after the model's own. */
struct batch_bind {
    int nb;
    layer_fn delta, attn, mtp;
    top_fn hidden, logits;
    long (*accepted)(struct rt *R, float *const *L, const long *DR, const long *AG); /* nb 2 to 4 */
};

struct model_desc {
    int n_layer, n_slots, split; /* split: half the rows on the host (link.h) */
    long d, vocab, ctx, frame_bytes;
    const unsigned char *full; /* per layer: 1 = attention */
    const struct tensor_bind *tensors;
    int n_tensors;
    const struct state_bind *states;
    int n_states;
    const struct batch_bind *batches; /* the largest first, the last one nb 1: a decode step */
    int n_batches;
    const float *(*embed)(struct rt *R, float *const *L, long tok);
    top_fn draft_logits; /* a draft's logits, over the common tokens */
    long (*draft_next)(struct rt *R, float *const *L, const float *X);
    long (*next)(struct rt *R, float *const *L, const float *X);
};

extern const struct model_desc MODEL;

/* The batch to take next: the largest within what is left of the prompt and within limit */
static inline int batch_for(long left, long limit)
{
    int i = 0;
    while (MODEL.batches[i].nb > left || MODEL.batches[i].nb > limit)
        i++;
    return i;
}

/* ---- the profile build (-DRT_PROFILE on model.c and engine.c, never on rt.c): every
 * call the generated code makes is timed where it is made. The scalars, the map loops
 * and the pointer arithmetic are what the table's total leaves over. */
enum {
    P_GEMV_BIG, P_GEMV_MID, P_GEMV_SMALL, P_GEMM, P_EW_TEAM, P_EW_ONE, P_SUM, P_MAXV, P_INDEX_OF, P_REDFIRST,
    P_IP21, P_IP12, P_OUTER11, P_SCALE0N, P_Q8_ROW, P_CAT, P_INDEX3, P_TRANSPOSE, P_MEMCPY,
    P_LINK_WAIT, P_LINK_COPY, P_LINK_DONE, P_N
};
extern uint64_t rt_prof_ticks[P_N], rt_prof_calls[P_N];

static inline uint64_t rt_prof_now(void)
{
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

#if defined(RT_PROFILE)
#define RT_TIMED(id, call)                                                                                             \
    do {                                                                                                               \
        const uint64_t t0_ = rt_prof_now();                                                                            \
        call;                                                                                                          \
        rt_prof_ticks[id] += rt_prof_now() - t0_;                                                                      \
        rt_prof_calls[id]++;                                                                                           \
    } while (0)
#define RT_TIMED_V(id, call)                                                                                           \
    __extension__({                                                                                                    \
        const uint64_t t0_ = rt_prof_now();                                                                            \
        __typeof__(call) v_ = call;                                                                                    \
        rt_prof_ticks[id] += rt_prof_now() - t0_;                                                                      \
        rt_prof_calls[id]++;                                                                                           \
        v_;                                                                                                            \
    })
#define rt_gemv(R, w, rows, cols, x, y)                                                                                \
    RT_TIMED((rows) >= 100000 ? P_GEMV_BIG : (rows) >= 512 ? P_GEMV_MID : P_GEMV_SMALL, (rt_gemv)(R, w, rows, cols, x, y))
#define rt_gemm(R, w, rows, cols, x, t, y) RT_TIMED(P_GEMM, (rt_gemm)(R, w, rows, cols, x, t, y))
#define rt_ew(R, op, z, a, b, s, n) RT_TIMED((n) >= 90000 ? P_EW_TEAM : P_EW_ONE, (rt_ew)(R, op, z, a, b, s, n))
#define rt_sum(a, n) RT_TIMED_V(P_SUM, (rt_sum)(a, n))
#define rt_maxv(a, n) RT_TIMED_V(P_MAXV, (rt_maxv)(a, n))
#define rt_index_of(a, n, v) RT_TIMED_V(P_INDEX_OF, (rt_index_of)(a, n, v))
#define rt_redfirst(z, a, r, c) RT_TIMED(P_REDFIRST, (rt_redfirst)(z, a, r, c))
#define rt_ip21(R, z, a, h, r, c, b) RT_TIMED(P_IP21, (rt_ip21)(R, z, a, h, r, c, b))
#define rt_ip12(R, z, a, h, t, c, b) RT_TIMED(P_IP12, (rt_ip12)(R, z, a, h, t, c, b))
#define rt_ip21_kv(R, z, kv, n1, c, l, h, t, b) RT_TIMED(P_IP21, (rt_ip21_kv)(R, z, kv, n1, c, l, h, t, b))
#define rt_ip12_kv(R, z, a, kv, n1, c, l, h, t) RT_TIMED(P_IP12, (rt_ip12_kv)(R, z, a, kv, n1, c, l, h, t))
#define rt_outer11(R, z, a, b, h, m, k) RT_TIMED(P_OUTER11, (rt_outer11)(R, z, a, b, h, m, k))
#define rt_outer_add(R, z, a, b, h, m, k) RT_TIMED(P_OUTER11, (rt_outer_add)(R, z, a, b, h, m, k))
#define rt_scale_outer(R, z, s, g, a, b, m, k) RT_TIMED(P_OUTER11, (rt_scale_outer)(R, z, s, g, a, b, m, k))
#define rt_gemv_go(R) RT_TIMED(P_GEMV_MID, (rt_gemv_go)(R))
#define rt_scale0n(R, z, a, b, h, c) RT_TIMED(P_SCALE0N, (rt_scale0n)(R, z, a, b, h, c))
#define rt_q8_row(w, c, r, z) RT_TIMED(P_Q8_ROW, (rt_q8_row)(w, c, r, z))
#define rt_cat(z, a, na, b, nb) RT_TIMED(P_CAT, (rt_cat)(z, a, na, b, nb))
#define rt_index3(R, z, a, n1, n2, l0, k0, l1, k1) RT_TIMED(P_INDEX3, (rt_index3)(R, z, a, n1, n2, l0, k0, l1, k1))
#define rt_transpose102(R, z, a, n0, n1, n2) RT_TIMED(P_TRANSPOSE, (rt_transpose102)(R, z, a, n0, n1, n2))
#define memcpy(d, s, n) RT_TIMED(P_MEMCPY, (memcpy)(d, s, n))
#endif

#endif
