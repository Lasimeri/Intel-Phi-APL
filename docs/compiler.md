# The compiler

`src/apl/compiler/` is an APL compiler written in APL, hosted by GNU APL, in the architecture of
Hsu's Co-dfns: the program is a table, and passes are whole-array operations over its columns. It
compiles a restricted dialect, for one model at a time, because shapes come from the model's tensors.

Co-dfns is AGPL-3.0: it was read for design only, and no code was copied.

```
scripts/engine.sh compile MODEL
  qwen35.apl + the model's tensor index (gguf_apl)
    -> out/gen/MODEL/model.c           the whole model on the card alone
    -> out/gen/MODEL/model_card.c      the card's half
    -> out/gen/MODEL/model_card_host.c the host's half
```

About 5 s per model (the 2B, measured). Generated C is never edited by hand.

## The passes

| File | Pass | Output |
|---|---|---|
| `tk.apl` | tokenizer: masks and scans over the whole source, no branches | `TKT TKS`: kind and text per token |
| `ps.apl` | parser: Bunda-Gerth pairwise reduction: each round every adjacent pair looks up its binding strength, and every pair stronger than both neighbours fuses, all at once | tree as an inverted table `PN_*` |
| `fl.apl` | flatten: one table in execution order; a function's meaning is the canonical text of its tree | `OP_F OP_V OP_A OP_B OP_X` |
| `in.apl` | inline: rounds until no calls remain; a function under the rank operator becomes a map region; names resolved by sorting, under GNU APL's dynamic scope | `I_*`, one entry |
| `sh.apl` | shapes and constant folding: folding a constant means executing it, here, in the interpreter. A rule table keyed on canonical text is the whitelist of primitives; a text with no rule is rejected | `S_*` |
| `sc.apl` | schedule: within what the data allow, gather each side's rows into as few phases as possible | permutes the table |
| `em.apl` | plan, placement, rounds, crossings, frames, C text | `E_*`, C |
| `fz.apl` | fusion | one loop per chain |
| `main.apl` | driver: bindings, tables, one entry per batch size, files | `.c` |

## The table

One row per operation, in execution order.

| Column | Meaning |
|---|---|
| `I_F` | the operation: canonical text, or `name num str elided strand index set setix map{ }map cell ret` |
| `I_V` | the value it produces |
| `I_A`, `I_B` | left and right argument value ids, ¯1 for none |
| `I_X` | name, literal or index ids; for `map{` the ranks |
| `I_I`, `I_R` | instance, and enclosing region (0 = none) |
| `S_K`, `S_V` | known at compile time, and its value |
| `S_S` | shape; ¯1 is `T`, the current token's positions so far, the only runtime extent |
| `S_I` | a runtime integer: `p` and what is computed from it |
| `E_VIEW` | writes no data: a pointer into another value |
| `E_BUF` | owns a place in the frame |
| `E_DEST`, `E_DESTI` | writes straight into a state (whole, or a row) |
| `E_FUSED` | a term of `a×s+b∘.×c`, taken lane by lane by the sum |
| `E_KV` | a history view: never built, the products read the KV cache |
| `E_SIDE` | 0 card, 1 host, 2 both (no data) |
| `E_GQ` | in a queued round; 2 marks the last |
| `E_PRIV` | the parallel region of a row, or ¯1 |
| `E_OFF`, `E_BASE` | frame offset; what a view ultimately reads |
| `E_XC`, `E_XH` | each side's crossing text before a row |
| `E_FZR`, `E_FZS`, `E_FZD` | a fused row's root; a root's rows; a root's map row, written in place |

## Placement

A two-state dynamic program over the table (`EMPlace`). A row is forced by what it is: a Q8_0 product goes to
the card; a state write, a scalar and a map's body go to the host. Each free row goes where it is cheapest, with
a change of side costing `EM_SWITCH_NS` (7,000 ns, what a crossing measured). `EMCost` holds the rough costs:
rows of `EM_BIG_LANES` (90,000) lanes or more go to the card's team, where a batch's arrays would otherwise cross.

## Rounds and read-ahead

Adjacent products on the card share one team round (`EMBatch`): a thread runs its share of each in turn with no
barrier between. A batch ends at the first card row that is not a product, at a change of side, or at a product
reading an earlier one's result.

A round that begins its phase opens before the wait for the host (`EMOpen`): the queue and `rt_gemv_open` go first,
and the threads read ahead into L2 through the time the card would idle. The compiler knows what comes next, so
nothing has to find out at run time.

## Frames

`EMAlloc` assigns places by last use, one pass over the table; a temporary lands on lines the last one warmed. A
matrix product's `x` and `y` are the exception: each has its own place with `EM_GAP` (4,096) bytes after it. Places in
contact cost the card about 5 % of its decode on the 2B (46.7 tok/s against 49.1), and reused ones about 8.7 %.

## Parallel maps

A map runs on the host's threads, one iteration each, when `EMCanSplit` proves it safe: an iteration writes no state
but its own row, its extents do not depend on a position, it is worth a team round (`EM_PAR_LANES`, 131,072 lanes in
all), and no row reaches that by itself. `EMOutline` writes it as a function, with temporaries per thread.

## Fusion

`fz.apl`. A chain of elementwise rows on the host becomes one loop: each row is a line of a per-lane sequence, the same
operations in the same order, with one store at the end. A value all of whose readers are in the chain never reaches
memory. This is `(C+A×B)÷D` as one loop, Hsu's scalar loop fusion, taken as far as it goes.

A row R joins the chain of root Q only if (`EMFzFits`): every reader of R is already in the chain, the shapes and region
match, and no state is written and no side crossed between R and Q. Roots are taken from the last row up, and the chain
grows to a fixed point, because a value read twice joins once both its readers have.

Besides elementwise rows:

- `+/` of a chain accumulates in the loop, as `rt_sum` does, when the lanes are a multiple of 8.
- `+⌿` of a list of a state's rows times a matrix (the depthwise conv) is one node, with its array of copies and its array
  of products never built (`EMFzIsTaps`).
- A map's cell goes straight into the map's row when it is the last thing the body makes, in whole 16-lane blocks
  (`EMFzInPlace`).
- `700⌊` under `exp` is an alias: `exp8` clamps its exponent already (`EMFzDeadMin`).

Because `EMFzPlan` changes who reads what and when, `EMAlloc` runs again after it: a fused row reads its inputs at the root.

Here is what one chain becomes: `G←W_A×SoftPlus W_DT+ALPHA[t;]` and `E←*G`, generated, abridged:

```c
for (long j = i0; j < i1; j += 8) {
    UNUSED const __m256 v0 = _mm256_load_ps(x0 + j);
    const __m256 r74 = _mm256_add_ps(v0, v1);
    const __m256 r76 = r74;                  /* 700⌊: an alias */
    const __m256 r77 = exp8(r76);
    const __m256 r79 = _mm256_add_ps(v3, r77);
    const __m256 r80 = log8(r79);
    const __m256 r82 = _mm256_mul_ps(v4, r80);
    const __m256 r165 = exp8(r82);
    _mm256_store_ps(z + j, r165);
}
```

The host builds with `-ffp-contract=off`, so this gives the same bits as the separate passes it replaced. That was
checked: 25 of 25 dump files identical.

## Compile-time settings

| Name | Value | Meaning |
|---|---|---|
| `EM_NBS` | 16 8 4 3 2 | batch sizes compiled, besides 1 |
| `EM_LOGIT_NB` | 4 | largest batch with logits, and with queued rounds |
| `WL`, `SL` | 8, 6 | ring and state levels: how far tokens can be taken back |
| `HOT` | 32768 | tokens a draft can be |
| `CTX` | 1024 | positions in the KV cache |
| `EM_BATCH` | 8 | most products in one round |
| `EM_BIG_LANES` | 90000 | lanes from which a free row runs on the card's team |
| `EM_PAR_LANES` | 131072 | lanes a map must reach to run on the host's threads |
| `EM_HOST_THREADS` | 14 | per-thread temporaries in a parallel map |
| `EM_SWITCH_NS` | 7000 | cost of a change of side |
| `EM_GAP` | 4096 | bytes after a product's `x` and `y` |

## Adding a primitive

1. Add its shape rule to `sh.apl`, keyed on the canonical text. Without one it is rejected, with its text.
2. Add its lowering to `EMRowText` in `em.apl`, and to `EMFzLine` in `fz.apl` if it is elementwise.
3. Add the C primitive to `src/card/rt.c`, `src/host/rt_host.c` or `src/host/vec8.h`, with its declaration in `rt.h`.
4. Run `tests/apl/parse_oracle.sh` and `shape_oracle.sh`, then `make check`.

The rule that decides where math goes: **model math is in `qwen35.apl`, and C holds only primitives, transport and I/O.**
A primitive is a machine form of an APL function, never a piece of the forward pass.
