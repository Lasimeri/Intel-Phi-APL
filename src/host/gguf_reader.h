/* gguf_reader.h: read a GGUF v3 header: metadata keys and the tensor table. Only
 * the header is read, never the tensor data. Shared by every host tool that
 * needs to know what a model file holds and where.
 * Type sizes: llama.cpp f5b9bd3 ggml/src/ggml.c type_traits and
 * ggml/src/ggml-common.h static_asserts. */
#ifndef APL_GGUF_READER_H
#define APL_GGUF_READER_H
#include <stddef.h>
#include <stdint.h>

/* GGUF value types. */
enum { GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32, GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64 };

/* The ggml tensor types the engine handles by id. */
enum { GGML_F32 = 0, GGML_Q8_0 = 8 };

struct gguf_key {
    char name[96];
    uint32_t type;      /* GGUF_* */
    uint32_t elem_type; /* for GGUF_ARR */
    uint64_t n;         /* array elements, or the full string length */
    char text[128];     /* the value as text; numeric arrays of up to 16 elements in full, others empty; strings cut */
};

struct gguf_tensor {
    char name[96];
    uint32_t n_dims;
    uint64_t dim[4];  /* ne0 first: the contiguous dimension */
    uint32_t type;    /* ggml type id */
    uint64_t offset;  /* absolute offset in the file */
    uint64_t bytes;
};

struct gguf {
    uint32_t version;
    uint64_t file_size;
    uint64_t header_end, data_start;
    uint64_t data_bytes; /* the sum of all tensors' bytes */
    uint64_t need;       /* the file size the tensor table requires */
    uint64_t n_keys, n_tensors;
    struct gguf_key *keys;
    struct gguf_tensor *tensors;
};

/* Reads the header of path into g. Returns 0, or -1 with a message in err. */
int gguf_load(struct gguf *g, const char *path, char *err, size_t err_cap);
void gguf_free(struct gguf *g);

const struct gguf_key *gguf_key(const struct gguf *g, const char *name);
const struct gguf_tensor *gguf_tensor(const struct gguf *g, const char *name);

/* The u32 value of a scalar key, or fallback if absent. */
uint64_t gguf_key_u64(const struct gguf *g, const char *name, uint64_t fallback);

const char *gguf_value_type_name(uint32_t type);
/* ggml type: its name, or NULL for an unknown id. */
const char *gguf_tensor_type_name(uint32_t type);
#define GGUF_MAX_TENSOR_TYPE 43

#endif
