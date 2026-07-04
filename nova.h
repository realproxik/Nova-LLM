// Copyright (c) NOVA AI. All rights reserved.

#ifndef NOVA_H
#define NOVA_H

#include "nova_config.h"
#include "nova_types.h"
#include "nova_tensor.h"
#include "nova_memory.h"
#include "nova_engine.h"
#include "nova_model.h"
#include "nova_tokenizer.h"
#include "nova_sampler.h"
#include "nova_attention.h"
#include "nova_mlp.h"
#include "nova_norm.h"
#include "nova_kv_cache.h"
#include "nova_paged_attn.h"
#include "nova_flash_attn.h"
#include "nova_moe.h"
#include "nova_quant.h"
#include "nova_gguf.h"
#include "nova_train.h"
#include "nova_optimizer.h"
#include "nova_loss.h"
#include "nova_data.h"
#include "nova_server.h"
#include "nova_cli.h"
#include "nova_download.h"
#include "nova_convert.h"
#include "nova_perplexity.h"
#include "nova_benchmark.h"
#include "nova_profile.h"
#include "nova_debug.h"
#include "nova_utils.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#ifdef NOVA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef NOVA_BUILD
#            define NOVA_API __declspec(dllexport)
#        else
#            define NOVA_API __declspec(dllimport)
#        endif
#    else
#        define NOVA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define NOVA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// VERSION
// ============================================================================

#define NOVA_VERSION_MAJOR 1
#define NOVA_VERSION_MINOR 0
#define NOVA_VERSION_PATCH 0
#define NOVA_VERSION_STRING "1.0.0"

#define NOVA_DEFAULT_SEED 0xFFFFFFFF
#define NOVA_TOKEN_NULL -1

// ============================================================================
// TYPES
// ============================================================================

typedef int32_t nova_pos;
typedef int32_t nova_token;
typedef int32_t nova_seq_id;

typedef struct nova_vocab nova_vocab_t;
typedef struct nova_model nova_model_t;
typedef struct nova_context nova_context_t;
typedef struct nova_sampler nova_sampler_t;
typedef struct nova_grammar nova_grammar_t;
typedef struct nova_lora nova_lora_t;
typedef struct nova_memory nova_memory_t;
typedef struct nova_engine nova_engine_t;
typedef struct nova_tensor nova_tensor_t;
typedef struct nova_kv_cache nova_kv_cache_t;
typedef struct nova_paged_attn nova_paged_attn_t;
typedef struct nova_flash_attn nova_flash_attn_t;
typedef struct nova_moe nova_moe_t;
typedef struct nova_quant nova_quant_t;
typedef struct nova_gguf nova_gguf_t;
typedef struct nova_trainer nova_trainer_t;
typedef struct nova_optimizer nova_optimizer_t;
typedef struct nova_loss nova_loss_t;
typedef struct nova_dataloader nova_dataloader_t;
typedef struct nova_server nova_server_t;

// ============================================================================
// ENUMS
// ============================================================================

enum nova_vocab_type {
    NOVA_VOCAB_TYPE_NONE   = 0,
    NOVA_VOCAB_TYPE_SPM    = 1,
    NOVA_VOCAB_TYPE_BPE    = 2,
    NOVA_VOCAB_TYPE_WPM    = 3,
    NOVA_VOCAB_TYPE_UGM    = 4,
    NOVA_VOCAB_TYPE_RWKV   = 5,
    NOVA_VOCAB_TYPE_PLAMO2 = 6,
};

enum nova_rope_type {
    NOVA_ROPE_TYPE_NONE   = -1,
    NOVA_ROPE_TYPE_NORM   = 0,
    NOVA_ROPE_TYPE_NEOX   = 1,
    NOVA_ROPE_TYPE_MROPE  = 2,
    NOVA_ROPE_TYPE_IMROPE = 3,
    NOVA_ROPE_TYPE_VISION = 4,
};

enum nova_token_type {
    NOVA_TOKEN_TYPE_UNDEFINED    = 0,
    NOVA_TOKEN_TYPE_NORMAL       = 1,
    NOVA_TOKEN_TYPE_UNKNOWN      = 2,
    NOVA_TOKEN_TYPE_CONTROL      = 3,
    NOVA_TOKEN_TYPE_USER_DEFINED = 4,
    NOVA_TOKEN_TYPE_UNUSED       = 5,
    NOVA_TOKEN_TYPE_BYTE         = 6,
};

enum nova_token_attr {
    NOVA_TOKEN_ATTR_UNDEFINED    = 0,
    NOVA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
    NOVA_TOKEN_ATTR_UNUSED       = 1 << 1,
    NOVA_TOKEN_ATTR_NORMAL       = 1 << 2,
    NOVA_TOKEN_ATTR_CONTROL      = 1 << 3,
    NOVA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    NOVA_TOKEN_ATTR_BYTE         = 1 << 5,
    NOVA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    NOVA_TOKEN_ATTR_LSTRIP       = 1 << 7,
    NOVA_TOKEN_ATTR_RSTRIP       = 1 << 8,
    NOVA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
};

enum nova_ftype {
    NOVA_FTYPE_ALL_F32              = 0,
    NOVA_FTYPE_MOSTLY_F16           = 1,
    NOVA_FTYPE_MOSTLY_Q4_0          = 2,
    NOVA_FTYPE_MOSTLY_Q4_1          = 3,
    NOVA_FTYPE_MOSTLY_Q8_0          = 4,
    NOVA_FTYPE_MOSTLY_Q5_0          = 5,
    NOVA_FTYPE_MOSTLY_Q5_1          = 6,
    NOVA_FTYPE_MOSTLY_Q2_K          = 7,
    NOVA_FTYPE_MOSTLY_Q3_K_S        = 8,
    NOVA_FTYPE_MOSTLY_Q3_K_M        = 9,
    NOVA_FTYPE_MOSTLY_Q3_K_L        = 10,
    NOVA_FTYPE_MOSTLY_Q4_K_S        = 11,
    NOVA_FTYPE_MOSTLY_Q4_K_M        = 12,
    NOVA_FTYPE_MOSTLY_Q5_K_S        = 13,
    NOVA_FTYPE_MOSTLY_Q5_K_M        = 14,
    NOVA_FTYPE_MOSTLY_Q6_K          = 15,
    NOVA_FTYPE_MOSTLY_IQ2_XXS       = 16,
    NOVA_FTYPE_MOSTLY_IQ2_XS        = 17,
    NOVA_FTYPE_MOSTLY_Q2_K_S        = 18,
    NOVA_FTYPE_MOSTLY_IQ3_XS        = 19,
    NOVA_FTYPE_MOSTLY_IQ3_XXS       = 20,
    NOVA_FTYPE_MOSTLY_IQ1_S         = 21,
    NOVA_FTYPE_MOSTLY_IQ4_NL        = 22,
    NOVA_FTYPE_MOSTLY_IQ3_S         = 23,
    NOVA_FTYPE_MOSTLY_IQ3_M         = 24,
    NOVA_FTYPE_MOSTLY_IQ2_S         = 25,
    NOVA_FTYPE_MOSTLY_IQ2_M         = 26,
    NOVA_FTYPE_MOSTLY_IQ4_XS        = 27,
    NOVA_FTYPE_MOSTLY_IQ1_M         = 28,
    NOVA_FTYPE_MOSTLY_BF16          = 29,
    NOVA_FTYPE_MOSTLY_NVFP4         = 30,
    NOVA_FTYPE_MOSTLY_Q1_0          = 31,
    NOVA_FTYPE_GUESSED              = 1024,
};

enum nova_rope_scaling_type {
    NOVA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    NOVA_ROPE_SCALING_TYPE_NONE        = 0,
    NOVA_ROPE_SCALING_TYPE_LINEAR      = 1,
    NOVA_ROPE_SCALING_TYPE_YARN        = 2,
    NOVA_ROPE_SCALING_TYPE_LONGROPE    = 3,
};

enum nova_pooling_type {
    NOVA_POOLING_TYPE_UNSPECIFIED = -1,
    NOVA_POOLING_TYPE_NONE = 0,
    NOVA_POOLING_TYPE_MEAN = 1,
    NOVA_POOLING_TYPE_CLS  = 2,
    NOVA_POOLING_TYPE_LAST = 3,
    NOVA_POOLING_TYPE_RANK = 4,
};

enum nova_attention_type {
    NOVA_ATTENTION_TYPE_UNSPECIFIED = -1,
    NOVA_ATTENTION_TYPE_CAUSAL      = 0,
    NOVA_ATTENTION_TYPE_NON_CAUSAL  = 1,
};

enum nova_split_mode {
    NOVA_SPLIT_MODE_NONE   = 0,
    NOVA_SPLIT_MODE_LAYER  = 1,
    NOVA_SPLIT_MODE_ROW    = 2,
    NOVA_SPLIT_MODE_TENSOR = 3,
};

enum nova_context_type {
    NOVA_CONTEXT_TYPE_DEFAULT = 0,
    NOVA_CONTEXT_TYPE_MTP     = 1,
};

enum nova_flash_attn_type {
    NOVA_FLASH_ATTN_TYPE_AUTO     = -1,
    NOVA_FLASH_ATTN_TYPE_DISABLED = 0,
    NOVA_FLASH_ATTN_TYPE_ENABLED  = 1,
};

enum nova_model_meta_key {
    NOVA_MODEL_META_KEY_SAMPLING_SEQUENCE,
    NOVA_MODEL_META_KEY_SAMPLING_TOP_K,
    NOVA_MODEL_META_KEY_SAMPLING_TOP_P,
    NOVA_MODEL_META_KEY_SAMPLING_MIN_P,
    NOVA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
    NOVA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
    NOVA_MODEL_META_KEY_SAMPLING_TEMP,
    NOVA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
    NOVA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
    NOVA_MODEL_META_KEY_SAMPLING_MIROSTAT,
    NOVA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
    NOVA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
};

enum nova_model_kv_override_type {
    NOVA_KV_OVERRIDE_TYPE_INT,
    NOVA_KV_OVERRIDE_TYPE_FLOAT,
    NOVA_KV_OVERRIDE_TYPE_BOOL,
    NOVA_KV_OVERRIDE_TYPE_STR,
};

// ============================================================================
// STRUCTS
// ============================================================================

typedef struct nova_token_data {
    nova_token id;
    float logit;
    float p;
} nova_token_data_t;

typedef struct nova_token_data_array {
    nova_token_data_t * data;
    size_t size;
    int64_t selected;
    bool sorted;
} nova_token_data_array_t;

typedef bool (*nova_progress_callback)(float progress, void * user_data);

typedef struct nova_batch {
    int32_t n_tokens;
    nova_token * token;
    float * embd;
    nova_pos * pos;
    int32_t * n_seq_id;
    nova_seq_id ** seq_id;
    int8_t * logits;
} nova_batch_t;

typedef struct nova_model_kv_override {
    enum nova_model_kv_override_type tag;
    char key[128];
    union {
        int64_t val_i64;
        double val_f64;
        bool val_bool;
        char val_str[128];
    };
} nova_model_kv_override_t;

typedef struct nova_model_params {
    int32_t n_gpu_layers;
    enum nova_split_mode split_mode;
    int32_t main_gpu;
    const float * tensor_split;
    nova_progress_callback progress_callback;
    void * progress_callback_user_data;
    const struct nova_model_kv_override * kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;
} nova_model_params_t;

typedef struct nova_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    enum nova_rope_scaling_type rope_scaling_type;
    enum nova_pooling_type pooling_type;
    enum nova_attention_type attention_type;
    enum nova_flash_attn_type flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
} nova_context_params_t;

typedef struct nova_sampler_chain_params {
    bool no_perf;
} nova_sampler_chain_params_t;

typedef struct nova_model_quantize_params {
    int32_t nthread;
    enum nova_ftype ftype;
    enum nova_tensor_type output_tensor_type;
    enum nova_tensor_type token_embedding_type;
    bool allow_requantize;
    bool quantize_output_tensor;
    bool only_copy;
    bool pure;
    bool keep_split;
    bool dry_run;
} nova_model_quantize_params_t;

typedef struct nova_logit_bias {
    nova_token token;
    float bias;
} nova_logit_bias_t;

typedef struct nova_chat_message {
    const char * role;
    const char * content;
} nova_chat_message_t;

// ============================================================================
// DEFAULT PARAMS
// ============================================================================

NOVA_API struct nova_model_params nova_model_default_params(void);
NOVA_API struct nova_context_params nova_context_default_params(void);
NOVA_API struct nova_sampler_chain_params nova_sampler_chain_default_params(void);
NOVA_API struct nova_model_quantize_params nova_model_quantize_default_params(void);

// ============================================================================
// BACKEND
// ============================================================================

NOVA_API void nova_backend_init(void);
NOVA_API void nova_backend_free(void);
NOVA_API void nova_numa_init(int numa_strategy);
NOVA_API const char * nova_print_system_info(void);
NOVA_API void nova_log_set(void (*callback)(const char * msg, void * user_data), void * user_data);

// ============================================================================
// MODEL
// ============================================================================

NOVA_API struct nova_model * nova_model_load_from_file(const char * path_model, struct nova_model_params params);
NOVA_API struct nova_model * nova_model_load_from_splits(const char ** paths, size_t n_paths, struct nova_model_params params);
NOVA_API void nova_model_save_to_file(const struct nova_model * model, const char * path_model);
NOVA_API void nova_model_free(struct nova_model * model);
NOVA_API void nova_model_free_weights(struct nova_model * model);

NOVA_API const struct nova_vocab * nova_model_get_vocab(const struct nova_model * model);
NOVA_API enum nova_rope_type nova_model_rope_type(const struct nova_model * model);

NOVA_API int32_t nova_model_n_ctx_train(const struct nova_model * model);
NOVA_API int32_t nova_model_n_embd(const struct nova_model * model);
NOVA_API int32_t nova_model_n_embd_inp(const struct nova_model * model);
NOVA_API int32_t nova_model_n_embd_out(const struct nova_model * model);
NOVA_API int32_t nova_model_n_layer(const struct nova_model * model);
NOVA_API int32_t nova_model_n_head(const struct nova_model * model);
NOVA_API int32_t nova_model_n_head_kv(const struct nova_model * model);
NOVA_API float nova_model_rope_freq_scale_train(const struct nova_model * model);
NOVA_API enum nova_ftype nova_model_ftype(const struct nova_model * model);
NOVA_API uint64_t nova_model_size(const struct nova_model * model);
NOVA_API uint64_t nova_model_n_params(const struct nova_model * model);
NOVA_API bool nova_model_has_encoder(const struct nova_model * model);
NOVA_API bool nova_model_has_decoder(const struct nova_model * model);
NOVA_API bool nova_model_is_recurrent(const struct nova_model * model);
NOVA_API bool nova_model_is_hybrid(const struct nova_model * model);
NOVA_API bool nova_model_is_diffusion(const struct nova_model * model);

NOVA_API int32_t nova_model_meta_val_str(const struct nova_model * model, const char * key, char * buf, size_t buf_size);
NOVA_API int32_t nova_model_meta_count(const struct nova_model * model);
NOVA_API const char * nova_model_meta_key_str(enum nova_model_meta_key key);
NOVA_API int32_t nova_model_meta_key_by_index(const struct nova_model * model, int32_t i, char * buf, size_t buf_size);
NOVA_API int32_t nova_model_meta_val_str_by_index(const struct nova_model * model, int32_t i, char * buf, size_t buf_size);
NOVA_API int32_t nova_model_desc(const struct nova_model * model, char * buf, size_t buf_size);
NOVA_API const char * nova_model_chat_template(const struct nova_model * model, const char * name);

NOVA_API uint32_t nova_model_quantize(const char * fname_inp, const char * fname_out, const nova_model_quantize_params_t * params);

// ============================================================================
// VOCAB
// ============================================================================

NOVA_API const char * nova_vocab_get_text(const struct nova_vocab * vocab, nova_token token);
NOVA_API float nova_vocab_get_score(const struct nova_vocab * vocab, nova_token token);
NOVA_API enum nova_token_attr nova_vocab_get_attr(const struct nova_vocab * vocab, nova_token token);
NOVA_API bool nova_vocab_is_eog(const struct nova_vocab * vocab, nova_token token);
NOVA_API bool nova_vocab_is_control(const struct nova_vocab * vocab, nova_token token);

NOVA_API nova_token nova_vocab_bos(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_eos(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_eot(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_sep(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_nl(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_pad(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_mask(const struct nova_vocab * vocab);
NOVA_API nova_token nova_vocab_cls(const struct nova_vocab * vocab);

NOVA_API bool nova_vocab_get_add_bos(const struct nova_vocab * vocab);
NOVA_API bool nova_vocab_get_add_eos(const struct nova_vocab * vocab);
NOVA_API bool nova_vocab_get_add_sep(const struct nova_vocab * vocab);

NOVA_API int32_t nova_vocab_n_tokens(const struct nova_vocab * vocab);
NOVA_API enum nova_vocab_type nova_vocab_type(const struct nova_vocab * vocab);

// ============================================================================
// CONTEXT
// ============================================================================

NOVA_API struct nova_context * nova_init_from_model(struct nova_model * model, struct nova_context_params params);
NOVA_API void nova_free(struct nova_context * ctx);
NOVA_API const struct nova_model * nova_get_model(const struct nova_context * ctx);
NOVA_API nova_memory_t nova_get_memory(const struct nova_context * ctx);

NOVA_API uint32_t nova_n_ctx(const struct nova_context * ctx);
NOVA_API uint32_t nova_n_batch(const struct nova_context * ctx);
NOVA_API uint32_t nova_n_ubatch(const struct nova_context * ctx);
NOVA_API uint32_t nova_n_seq_max(const struct nova_context * ctx);

NOVA_API void nova_set_n_threads(struct nova_context * ctx, int32_t n_threads, int32_t n_threads_batch);
NOVA_API int32_t nova_n_threads(struct nova_context * ctx);
NOVA_API int32_t nova_n_threads_batch(struct nova_context * ctx);

NOVA_API void nova_set_embeddings(struct nova_context * ctx, bool embeddings);
NOVA_API void nova_set_causal_attn(struct nova_context * ctx, bool causal_attn);
NOVA_API void nova_synchronize(struct nova_context * ctx);

// ============================================================================
// BATCH
// ============================================================================

NOVA_API struct nova_batch nova_batch_get_one(nova_token * tokens, int32_t n_tokens);
NOVA_API struct nova_batch nova_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
NOVA_API void nova_batch_free(struct nova_batch batch);

// ============================================================================
// ENCODE / DECODE
// ============================================================================

NOVA_API int32_t nova_encode(struct nova_context * ctx, struct nova_batch batch);
NOVA_API int32_t nova_decode(struct nova_context * ctx, struct nova_batch batch);

// ============================================================================
// LOGITS & EMBEDDINGS
// ============================================================================

NOVA_API float * nova_get_logits(struct nova_context * ctx);
NOVA_API float * nova_get_logits_ith(struct nova_context * ctx, int32_t i);
NOVA_API float * nova_get_embeddings(struct nova_context * ctx);
NOVA_API float * nova_get_embeddings_ith(struct nova_context * ctx, int32_t i);
NOVA_API float * nova_get_embeddings_seq(struct nova_context * ctx, nova_seq_id seq_id);
NOVA_API enum nova_pooling_type nova_pooling_type(const struct nova_context * ctx);

// ============================================================================
// STATE / SESSIONS
// ============================================================================

NOVA_API size_t nova_state_get_size(struct nova_context * ctx);
NOVA_API size_t nova_state_get_data(struct nova_context * ctx, uint8_t * dst, size_t size);
NOVA_API size_t nova_state_set_data(struct nova_context * ctx, const uint8_t * src, size_t size);
NOVA_API bool nova_state_load_file(struct nova_context * ctx, const char * path_session, nova_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
NOVA_API bool nova_state_save_file(struct nova_context * ctx, const char * path_session, const nova_token * tokens, size_t n_token_count);

NOVA_API size_t nova_state_seq_get_size(struct nova_context * ctx, nova_seq_id seq_id);
NOVA_API size_t nova_state_seq_get_data(struct nova_context * ctx, uint8_t * dst, size_t size, nova_seq_id seq_id);
NOVA_API size_t nova_state_seq_set_data(struct nova_context * ctx, const uint8_t * src, size_t size, nova_seq_id dest_seq_id);
NOVA_API size_t nova_state_seq_save_file(struct nova_context * ctx, const char * filepath, nova_seq_id seq_id, const nova_token * tokens, size_t n_token_count);
NOVA_API size_t nova_state_seq_load_file(struct nova_context * ctx, const char * filepath, nova_seq_id dest_seq_id, nova_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);

// ============================================================================
// MEMORY
// ============================================================================

NOVA_API void nova_memory_clear(nova_memory_t mem, bool data);
NOVA_API bool nova_memory_seq_rm(nova_memory_t mem, nova_seq_id seq_id, nova_pos p0, nova_pos p1);
NOVA_API void nova_memory_seq_cp(nova_memory_t mem, nova_seq_id seq_id_src, nova_seq_id seq_id_dst, nova_pos p0, nova_pos p1);
NOVA_API void nova_memory_seq_keep(nova_memory_t mem, nova_seq_id seq_id);
NOVA_API void nova_memory_seq_add(nova_memory_t mem, nova_seq_id seq_id, nova_pos p0, nova_pos p1, nova_pos delta);
NOVA_API void nova_memory_seq_div(nova_memory_t mem, nova_seq_id seq_id, nova_pos p0, nova_pos p1, int d);
NOVA_API nova_pos nova_memory_seq_pos_min(nova_memory_t mem, nova_seq_id seq_id);
NOVA_API nova_pos nova_memory_seq_pos_max(nova_memory_t mem, nova_seq_id seq_id);
NOVA_API bool nova_memory_can_shift(nova_memory_t mem);

// ============================================================================
// TOKENIZATION
// ============================================================================

NOVA_API int32_t nova_tokenize(const struct nova_vocab * vocab, const char * text, int32_t text_len, nova_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
NOVA_API int32_t nova_token_to_piece(const struct nova_vocab * vocab, nova_token token, char * buf, int32_t length, int32_t lstrip, bool special);
NOVA_API int32_t nova_detokenize(const struct nova_vocab * vocab, const nova_token * tokens, int32_t n_tokens, char * text, int32_t text_len_max, bool remove_special, bool unparse_special);

// ============================================================================
// CHAT TEMPLATES
// ============================================================================

NOVA_API int32_t nova_chat_apply_template(const char * tmpl, const struct nova_chat_message * chat, size_t n_msg, bool add_ass, char * buf, int32_t length);
NOVA_API int32_t nova_chat_builtin_templates(const char ** output, size_t len);

// ============================================================================
// SAMPLER
// ============================================================================

typedef void * nova_sampler_context_t;

struct nova_sampler_data {
    struct nova_tensor * logits;
    struct nova_tensor * probs;
    struct nova_tensor * sampled;
    struct nova_tensor * candidates;
};

struct nova_sampler_i {
    const char * (*name)(const struct nova_sampler * smpl);
    void (*accept)(struct nova_sampler * smpl, nova_token token);
    void (*apply)(struct nova_sampler * smpl, nova_token_data_array_t * cur_p);
    void (*reset)(struct nova_sampler * smpl);
    struct nova_sampler * (*clone)(const struct nova_sampler * smpl);
    void (*free)(struct nova_sampler * smpl);
    bool (*backend_init)(struct nova_sampler * smpl, void * buft);
    void (*backend_accept)(struct nova_sampler * smpl, void * ctx, void * gf, void * selected_token);
    void (*backend_apply)(struct nova_sampler * smpl, void * ctx, void * gf, struct nova_sampler_data * data);
    void (*backend_set_input)(struct nova_sampler * smpl);
};

struct nova_sampler {
    struct nova_sampler_i * iface;
    nova_sampler_context_t ctx;
};

NOVA_API struct nova_sampler * nova_sampler_init(struct nova_sampler_i * iface, nova_sampler_context_t ctx);
NOVA_API const char * nova_sampler_name(const struct nova_sampler * smpl);
NOVA_API void nova_sampler_accept(struct nova_sampler * smpl, nova_token token);
NOVA_API void nova_sampler_apply(struct nova_sampler * smpl, nova_token_data_array_t * cur_p);
NOVA_API void nova_sampler_reset(struct nova_sampler * smpl);
NOVA_API struct nova_sampler * nova_sampler_clone(const struct nova_sampler * smpl);
NOVA_API void nova_sampler_free(struct nova_sampler * smpl);
NOVA_API nova_token nova_sampler_sample(struct nova_sampler * smpl, struct nova_context * ctx, int32_t idx);
NOVA_API uint32_t nova_sampler_get_seed(const struct nova_sampler * smpl);

// ============================================================================
// SAMPLER - CHAIN
// ============================================================================

NOVA_API struct nova_sampler * nova_sampler_chain_init(struct nova_sampler_chain_params params);
NOVA_API void nova_sampler_chain_add(struct nova_sampler * chain, struct nova_sampler * smpl);
NOVA_API struct nova_sampler * nova_sampler_chain_get(struct nova_sampler * chain, int32_t i);
NOVA_API int nova_sampler_chain_n(const struct nova_sampler * chain);
NOVA_API struct nova_sampler * nova_sampler_chain_remove(struct nova_sampler * chain, int32_t i);

// ============================================================================
// SAMPLER - INIT
// ============================================================================

NOVA_API struct nova_sampler * nova_sampler_init_greedy(void);
NOVA_API struct nova_sampler * nova_sampler_init_dist(uint32_t seed);
NOVA_API struct nova_sampler * nova_sampler_init_top_k(int32_t k);
NOVA_API struct nova_sampler * nova_sampler_init_top_p(float p, size_t min_keep);
NOVA_API struct nova_sampler * nova_sampler_init_min_p(float p, size_t min_keep);
NOVA_API struct nova_sampler * nova_sampler_init_typical(float p, size_t min_keep);
NOVA_API struct nova_sampler * nova_sampler_init_temp(float t);
NOVA_API struct nova_sampler * nova_sampler_init_temp_ext(float t, float delta, float exponent);
NOVA_API struct nova_sampler * nova_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed);
NOVA_API struct nova_sampler * nova_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m);
NOVA_API struct nova_sampler * nova_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta);
NOVA_API struct nova_sampler * nova_sampler_init_grammar(const struct nova_vocab * vocab, const char * grammar_str, const char * grammar_root);
NOVA_API struct nova_sampler * nova_sampler_init_penalties(int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);
NOVA_API struct nova_sampler * nova_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias, const nova_logit_bias_t * logit_bias);
NOVA_API struct nova_sampler * nova_sampler_init_infill(const struct nova_vocab * vocab);

// ============================================================================
// LORA
// ============================================================================

NOVA_API struct nova_lora * nova_lora_init(struct nova_model * model, const char * path_lora);
NOVA_API void nova_lora_free(struct nova_lora * lora);
NOVA_API int32_t nova_lora_meta_val_str(const struct nova_lora * lora, const char * key, char * buf, size_t buf_size);
NOVA_API int32_t nova_lora_meta_count(const struct nova_lora * lora);
NOVA_API int32_t nova_lora_meta_key_by_index(const struct nova_lora * lora, int32_t i, char * buf, size_t buf_size);
NOVA_API int32_t nova_lora_meta_val_str_by_index(const struct nova_lora * lora, int32_t i, char * buf, size_t buf_size);
NOVA_API int32_t nova_set_adapters_lora(struct nova_context * ctx, struct nova_lora ** adapters, size_t n_adapters, float * scales);

// ============================================================================
// SPLIT
// ============================================================================

NOVA_API int32_t nova_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);
NOVA_API int32_t nova_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

// ============================================================================
// PERFORMANCE
// ============================================================================

struct nova_perf_context_data {
    double t_start_ms;
    double t_load_ms;
    double t_p_eval_ms;
    double t_eval_ms;
    int32_t n_p_eval;
    int32_t n_eval;
    int32_t n_reused;
};

struct nova_perf_sampler_data {
    double t_sample_ms;
    int32_t n_sample;
};

NOVA_API struct nova_perf_context_data nova_perf_context(const struct nova_context * ctx);
NOVA_API void nova_perf_context_print(const struct nova_context * ctx);
NOVA_API void nova_perf_context_reset(struct nova_context * ctx);
NOVA_API struct nova_perf_sampler_data nova_perf_sampler(const struct nova_sampler * chain);
NOVA_API void nova_perf_sampler_print(const struct nova_sampler * chain);
NOVA_API void nova_perf_sampler_reset(struct nova_sampler * chain);

// ============================================================================
// TRAINING
// ============================================================================

typedef bool (*nova_opt_param_filter)(const struct nova_tensor * tensor, void * userdata);
NOVA_API bool nova_opt_param_filter_all(const struct nova_tensor * tensor, void * userdata);

struct nova_opt_params {
    uint32_t n_ctx_train;
    nova_opt_param_filter param_filter;
    void * param_filter_ud;
    void * get_opt_pars;
    void * get_opt_pars_ud;
    int optimizer_type;
};

NOVA_API void nova_opt_init(struct nova_context * lctx, struct nova_model * model, struct nova_opt_params lopt_params);
NOVA_API void nova_opt_epoch(struct nova_context * lctx, void * dataset, void * result_train, void * result_eval, int64_t idata_split, void * callback_train, void * callback_eval);

// ============================================================================
// HELPERS
// ============================================================================

NOVA_API int64_t nova_time_us(void);
NOVA_API size_t nova_max_devices(void);
NOVA_API size_t nova_max_parallel_sequences(void);
NOVA_API bool nova_supports_mmap(void);
NOVA_API bool nova_supports_mlock(void);
NOVA_API bool nova_supports_gpu_offload(void);

// ============================================================================
// DEPRECATED
// ============================================================================

DEPRECATED(NOVA_API struct nova_model * nova_load_model_from_file(const char * path_model, struct nova_model_params params), "use nova_model_load_from_file instead");
DEPRECATED(NOVA_API struct nova_context * nova_new_context_with_model(struct nova_model * model, struct nova_context_params params), "use nova_init_from_model instead");
DEPRECATED(NOVA_API void nova_free_model(struct nova_model * model), "use nova_model_free instead");
DEPRECATED(NOVA_API int32_t nova_n_ctx_train(const struct nova_model * model), "use nova_model_n_ctx_train instead");
DEPRECATED(NOVA_API int32_t nova_n_embd(const struct nova_model * model), "use nova_model_n_embd instead");
DEPRECATED(NOVA_API int32_t nova_n_layer(const struct nova_model * model), "use nova_model_n_layer instead");
DEPRECATED(NOVA_API int32_t nova_n_head(const struct nova_model * model), "use nova_model_n_head instead");
DEPRECATED(NOVA_API int32_t nova_n_vocab(const struct nova_vocab * vocab), "use nova_vocab_n_tokens instead");
DEPRECATED(NOVA_API const char * nova_token_get_text(const struct nova_vocab * vocab, nova_token token), "use nova_vocab_get_text instead");
DEPRECATED(NOVA_API float nova_token_get_score(const struct nova_vocab * vocab, nova_token token), "use nova_vocab_get_score instead");
DEPRECATED(NOVA_API bool nova_token_is_eog(const struct nova_vocab * vocab, nova_token token), "use nova_vocab_is_eog instead");
DEPRECATED(NOVA_API nova_token nova_token_bos(const struct nova_vocab * vocab), "use nova_vocab_bos instead");
DEPRECATED(NOVA_API nova_token nova_token_eos(const struct nova_vocab * vocab), "use nova_vocab_eos instead");
DEPRECATED(NOVA_API size_t nova_get_state_size(struct nova_context * ctx), "use nova_state_get_size instead");
DEPRECATED(NOVA_API size_t nova_copy_state_data(struct nova_context * ctx, uint8_t * dst), "use nova_state_get_data instead");
DEPRECATED(NOVA_API size_t nova_set_state_data(struct nova_context * ctx, const uint8_t * src), "use nova_state_set_data instead");
DEPRECATED(NOVA_API bool nova_load_session_file(struct nova_context * ctx, const char * path_session, nova_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out), "use nova_state_load_file instead");
DEPRECATED(NOVA_API bool nova_save_session_file(struct nova_context * ctx, const char * path_session, const nova_token * tokens, size_t n_token_count), "use nova_state_save_file instead");
DEPRECATED(NOVA_API struct nova_sampler * nova_sampler_init_grammar_lazy(const struct nova_vocab * vocab, const char * grammar_str, const char * grammar_root, const char ** trigger_words, size_t num_trigger_words, const nova_token * trigger_tokens, size_t num_trigger_tokens), "use nova_sampler_init_grammar_lazy_patterns instead");

// ============================================================================
// CUSTOM SAMPLERS
// ============================================================================

NOVA_API struct nova_sampler * nova_sampler_init_top_n_sigma(float n);
NOVA_API struct nova_sampler * nova_sampler_init_dry(const struct nova_vocab * vocab, int32_t n_ctx_train, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char ** seq_breakers, size_t num_breakers);
NOVA_API struct nova_sampler * nova_sampler_init_adaptive_p(float target, float decay, uint32_t seed);
NOVA_API struct nova_sampler * nova_sampler_init_grammar_lazy_patterns(const struct nova_vocab * vocab, const char * grammar_str, const char * grammar_root, const char ** trigger_patterns, size_t num_trigger_patterns, const nova_token * trigger_tokens, size_t num_trigger_tokens);

// ============================================================================
// NOVA TENSOR API
// ============================================================================

NOVA_API struct nova_tensor * nova_tensor_create(size_t rows, size_t cols, enum nova_tensor_type dtype);
NOVA_API void nova_tensor_free(struct nova_tensor * t);
NOVA_API void nova_tensor_zero(struct nova_tensor * t);
NOVA_API void nova_tensor_copy(struct nova_tensor * dst, struct nova_tensor * src);
NOVA_API void nova_tensor_add(struct nova_tensor * a, struct nova_tensor * b, struct nova_tensor * out);
NOVA_API void nova_tensor_mul(struct nova_tensor * a, struct nova_tensor * b, struct nova_tensor * out);
NOVA_API void nova_tensor_matmul(struct nova_tensor * a, struct nova_tensor * b, struct nova_tensor * out);
NOVA_API void nova_tensor_softmax(struct nova_tensor * t);
NOVA_API void nova_tensor_rms_norm(struct nova_tensor * t, struct nova_tensor * weight, float eps);
NOVA_API void nova_tensor_layer_norm(struct nova_tensor * t, struct nova_tensor * weight, struct nova_tensor * bias, float eps);
NOVA_API void nova_tensor_gelu(struct nova_tensor * t);
NOVA_API void nova_tensor_silu(struct nova_tensor * t);
NOVA_API void nova_tensor_relu(struct nova_tensor * t);
NOVA_API void nova_tensor_dropout(struct nova_tensor * t, float p);
NOVA_API void nova_tensor_quantize(struct nova_tensor * t, struct nova_tensor * out, int bits);
NOVA_API void nova_tensor_dequantize(struct nova_tensor * t, struct nova_tensor * out);
NOVA_API void nova_tensor_print(struct nova_tensor * t, int max_items);
NOVA_API size_t nova_tensor_size(struct nova_tensor * t);
NOVA_API size_t nova_tensor_bytes(struct nova_tensor * t);
NOVA_API float nova_tensor_mean(struct nova_tensor * t);
NOVA_API float nova_tensor_std(struct nova_tensor * t);
NOVA_API float nova_tensor_max(struct nova_tensor * t);
NOVA_API float nova_tensor_min(struct nova_tensor * t);

// ============================================================================
// NOVA KV CACHE API
// ============================================================================

NOVA_API struct nova_kv_cache * nova_kv_cache_create(size_t num_heads, size_t head_dim, size_t max_seq, size_t max_batch);
NOVA_API void nova_kv_cache_free(struct nova_kv_cache * cache);
NOVA_API void nova_kv_cache_clear(struct nova_kv_cache * cache);
NOVA_API void nova_kv_cache_append(struct nova_kv_cache * cache, const float * k, const float * v, size_t seq_len);
NOVA_API float * nova_kv_cache_get_k(struct nova_kv_cache * cache, size_t pos);
NOVA_API float * nova_kv_cache_get_v(struct nova_kv_cache * cache, size_t pos);
NOVA_API size_t nova_kv_cache_get_seq_len(struct nova_kv_cache * cache);
NOVA_API size_t nova_kv_cache_get_memory_usage(struct nova_kv_cache * cache);

// ============================================================================
// NOVA PAGED ATTENTION API
// ============================================================================

NOVA_API struct nova_paged_attn * nova_paged_attn_create(size_t num_heads, size_t head_dim, size_t block_size, size_t max_blocks);
NOVA_API void nova_paged_attn_free(struct nova_paged_attn * paged);
NOVA_API void nova_paged_attn_clear(struct nova_paged_attn * paged);
NOVA_API uint64_t nova_paged_attn_allocate_page(struct nova_paged_attn * paged);
NOVA_API void nova_paged_attn_free_page(struct nova_paged_attn * paged, uint64_t page_id);
NOVA_API void nova_paged_attn_set_k(struct nova_paged_attn * paged, uint64_t page_id, size_t pos, const float * k, size_t len);
NOVA_API void nova_paged_attn_set_v(struct nova_paged_attn * paged, uint64_t page_id, size_t pos, const float * v, size_t len);
NOVA_API const float * nova_paged_attn_get_k(struct nova_paged_attn * paged, uint64_t page_id, size_t pos);
NOVA_API const float * nova_paged_attn_get_v(struct nova_paged_attn * paged, uint64_t page_id, size_t pos);
NOVA_API size_t nova_paged_attn_get_num_pages(struct nova_paged_attn * paged);
NOVA_API size_t nova_paged_attn_get_used_blocks(struct nova_paged_attn * paged);
NOVA_API size_t nova_paged_attn_get_free_blocks(struct nova_paged_attn * paged);
NOVA_API void nova_paged_attn_defragment(struct nova_paged_attn * paged);

// ============================================================================
// NOVA FLASH ATTENTION API
// ============================================================================

NOVA_API struct nova_flash_attn * nova_flash_attn_create(size_t num_heads, size_t head_dim, size_t block_size);
NOVA_API void nova_flash_attn_free(struct nova_flash_attn * flash);
NOVA_API void nova_flash_attn_forward(struct nova_flash_attn * flash, const float * Q, const float * K, const float * V, float * O, size_t batch, size_t seq_len, bool causal, float scale);
NOVA_API void nova_flash_attn_backward(struct nova_flash_attn * flash, const float * dO, const float * Q, const float * K, const float * V, const float * O, float * dQ, float * dK, float * dV, size_t batch, size_t seq_len);

// ============================================================================
// NOVA MOE API
// ============================================================================

NOVA_API struct nova_moe * nova_moe_create(size_t num_experts, size_t top_k, size_t hidden_size, size_t intermediate_size);
NOVA_API void nova_moe_free(struct nova_moe * moe);
NOVA_API void nova_moe_set_expert_weights(struct nova_moe * moe, size_t expert_id, const float * w1, const float * w2, size_t size);
NOVA_API void nova_moe_set_router_weights(struct nova_moe * moe, const float * router, size_t size);
NOVA_API void nova_moe_forward(struct nova_moe * moe, const float * input, float * output, size_t batch, size_t seq_len, size_t hidden_size);
NOVA_API void nova_moe_forward_with_aux_loss(struct nova_moe * moe, const float * input, float * output, float * aux_loss, size_t batch, size_t seq_len, size_t hidden_size);
NOVA_API void nova_moe_backward(struct nova_moe * moe, const float * grad_output, float * grad_input, float * grad_router, size_t batch, size_t seq_len, size_t hidden_size);
NOVA_API size_t nova_moe_get_num_experts(struct nova_moe * moe);
NOVA_API size_t nova_moe_get_top_k(struct nova_moe * moe);

// ============================================================================
// NOVA QUANT API
// ============================================================================

NOVA_API struct nova_quant * nova_quant_create(void);
NOVA_API void nova_quant_free(struct nova_quant * quant);
NOVA_API void nova_quant_int8(const float * input, int8_t * output, float * scale, size_t size);
NOVA_API void nova_quant_int8_per_tensor(const float * input, int8_t * output, float * scale, size_t rows, size_t cols);
NOVA_API void nova_dequant_int8(const int8_t * input, float * output, float scale, size_t size);
NOVA_API void nova_dequant_int8_per_tensor(const int8_t * input, float * output, const float * scale, size_t size);
NOVA_API void nova_quant_int4(const float * input, uint8_t * output, float * scale, size_t size);
NOVA_API void nova_dequant_int4(const uint8_t * input, float * output, float scale, size_t size);
NOVA_API void nova_quant_fp8(const float * input, uint8_t * output, size_t size);
NOVA_API void nova_dequant_fp8(const uint8_t * input, float * output, size_t size);
NOVA_API void nova_quant_gptq(const float * input, int8_t * output, float * scale, int * zero_point, size_t rows, size_t cols, size_t group_size);
NOVA_API void nova_dequant_gptq(const int8_t * input, float * output, const float * scale, const int * zero_point, size_t rows, size_t cols, size_t group_size);

// ============================================================================
// NOVA GGUF API
// ============================================================================

NOVA_API struct nova_gguf * nova_gguf_load(const char * path);
NOVA_API void nova_gguf_free(struct nova_gguf * gguf);
NOVA_API int nova_gguf_get_version(struct nova_gguf * gguf);
NOVA_API int nova_gguf_get_tensor_count(struct nova_gguf * gguf);
NOVA_API size_t nova_gguf_get_total_size(struct nova_gguf * gguf);
NOVA_API const char * nova_gguf_get_metadata(struct nova_gguf * gguf, const char * key);
NOVA_API const char * nova_gguf_get_tensor_name(struct nova_gguf * gguf, int idx);
NOVA_API size_t nova_gguf_get_tensor_size(struct nova_gguf * gguf, int idx);
NOVA_API int nova_gguf_get_tensor_type(struct nova_gguf * gguf, int idx);
NOVA_API int nova_gguf_load_tensor(struct nova_gguf * gguf, const char * name, float * buffer, size_t size);
NOVA_API const char * nova_gguf_get_chat_template(struct nova_gguf * gguf);
NOVA_API int nova_gguf_has_tensor(struct nova_gguf * gguf, const char * name);

// ============================================================================
// NOVA ENGINE API
// ============================================================================

NOVA_API struct nova_engine * nova_engine_create(void);
NOVA_API void nova_engine_free(struct nova_engine * engine);
NOVA_API int nova_engine_init(struct nova_engine * engine, const char * model_path);
NOVA_API int nova_engine_load_weights(struct nova_engine * engine, const char * path);
NOVA_API int nova_engine_unload_weights(struct nova_engine * engine);
NOVA_API int nova_engine_forward(struct nova_engine * engine, const float * input, float * output, size_t batch, size_t seq_len);
NOVA_API int nova_engine_generate(struct nova_engine * engine, const int * tokens, int * output, size_t n_tokens, int max_new, float temp, float top_p, int top_k);
NOVA_API int nova_engine_generate_stream(struct nova_engine * engine, const int * tokens, size_t n_tokens, void (*callback)(int token, float prob), int max_new, float temp, float top_p, int top_k);
NOVA_API void nova_engine_reset(struct nova_engine * engine);
NOVA_API void nova_engine_set_temperature(struct nova_engine * engine, float temp);
NOVA_API void nova_engine_set_top_p(struct nova_engine * engine, float top_p);
NOVA_API void nova_engine_set_top_k(struct nova_engine * engine, int top_k);
NOVA_API void nova_engine_set_max_tokens(struct nova_engine * engine, int max_tokens);
NOVA_API int nova_engine_is_loaded(struct nova_engine * engine);
NOVA_API size_t nova_engine_get_memory_usage(struct nova_engine * engine);
NOVA_API size_t nova_engine_get_cache_usage(struct nova_engine * engine);
NOVA_API struct nova_model * nova_engine_get_model(struct nova_engine * engine);

// ============================================================================
// NOVA TRAINER API
// ============================================================================

NOVA_API struct nova_trainer * nova_trainer_create(void);
NOVA_API void nova_trainer_free(struct nova_trainer * trainer);
NOVA_API void nova_trainer_init(struct nova_trainer * trainer, struct nova_model * model, struct nova_optimizer * optimizer, struct nova_dataloader * dataloader, struct nova_loss * loss);
NOVA_API void nova_trainer_train(struct nova_trainer * trainer);
NOVA_API void nova_trainer_train_step(struct nova_trainer * trainer);
NOVA_API void nova_trainer_eval(struct nova_trainer * trainer);
NOVA_API void nova_trainer_save_checkpoint(struct nova_trainer * trainer, const char * path);
NOVA_API void nova_trainer_load_checkpoint(struct nova_trainer * trainer, const char * path);
NOVA_API void nova_trainer_set_progress_callback(struct nova_trainer * trainer, void (*callback)(float loss, int step, float lr));
NOVA_API void nova_trainer_set_early_stopping(struct nova_trainer * trainer, float patience, float min_delta);
NOVA_API void nova_trainer_stop(struct nova_trainer * trainer);
NOVA_API int nova_trainer_is_training(struct nova_trainer * trainer);
NOVA_API struct nova_trainer_metrics nova_trainer_get_metrics(struct nova_trainer * trainer);

// ============================================================================
// NOVA OPTIMIZER API
// ============================================================================

NOVA_API struct nova_optimizer * nova_optimizer_adamw_create(float lr, float beta1, float beta2, float eps, float weight_decay);
NOVA_API struct nova_optimizer * nova_optimizer_sgd_create(float lr, float momentum, float weight_decay);
NOVA_API void nova_optimizer_free(struct nova_optimizer * opt);
NOVA_API void nova_optimizer_init(struct nova_optimizer * opt);
NOVA_API void nova_optimizer_step(struct nova_optimizer * opt);
NOVA_API void nova_optimizer_zero_grad(struct nova_optimizer * opt);
NOVA_API void nova_optimizer_add_param(struct nova_optimizer * opt, float * param, float * grad, size_t size);
NOVA_API void nova_optimizer_set_lr(struct nova_optimizer * opt, float lr);
NOVA_API float nova_optimizer_get_lr(struct nova_optimizer * opt);
NOVA_API void nova_optimizer_save(struct nova_optimizer * opt, const char * path);
NOVA_API void nova_optimizer_load(struct nova_optimizer * opt, const char * path);

// ============================================================================
// NOVA LOSS API
// ============================================================================

NOVA_API struct nova_loss * nova_loss_cross_entropy_create(float label_smoothing, float ignore_index);
NOVA_API struct nova_loss * nova_loss_mse_create(float reduction);
NOVA_API struct nova_loss * nova_loss_kl_divergence_create(int log_target);
NOVA_API void nova_loss_free(struct nova_loss * loss);
NOVA_API float nova_loss_compute(struct nova_loss * loss, const float * predictions, const float * targets, size_t size);
NOVA_API void nova_loss_gradient(struct nova_loss * loss, float * grad);
NOVA_API void nova_loss_reset(struct nova_loss * loss);

// ============================================================================
// NOVA DATALOADER API
// ============================================================================

NOVA_API struct nova_dataloader * nova_dataloader_create(const char * data_path, size_t batch_size, size_t seq_len, int shuffle);
NOVA_API void nova_dataloader_free(struct nova_dataloader * dataloader);
NOVA_API void nova_dataloader_reset(struct nova_dataloader * dataloader);
NOVA_API int nova_dataloader_has_next(struct nova_dataloader * dataloader);
NOVA_API void nova_dataloader_next_batch(struct nova_dataloader * dataloader, float * inputs, float * targets);
NOVA_API void nova_dataloader_shuffle(struct nova_dataloader * dataloader);
NOVA_API size_t nova_dataloader_get_batch_size(struct nova_dataloader * dataloader);
NOVA_API size_t nova_dataloader_get_num_batches(struct nova_dataloader * dataloader);
NOVA_API size_t nova_dataloader_get_dataset_size(struct nova_dataloader * dataloader);
NOVA_API void nova_dataloader_get_eval_data(struct nova_dataloader * dataloader, float * inputs, float * targets, size_t num_batches);

// ============================================================================
// NOVA SERVER API
// ============================================================================

NOVA_API struct nova_server * nova_server_create(int port, int workers);
NOVA_API void nova_server_free(struct nova_server * server);
NOVA_API void nova_server_start(struct nova_server * server, struct nova_model * model);
NOVA_API void nova_server_stop(struct nova_server * server);
NOVA_API void nova_server_set_completion_handler(struct nova_server * server, void (*handler)(const char * req, char * resp, size_t resp_size));
NOVA_API void nova_server_set_chat_handler(struct nova_server * server, void (*handler)(const char * req, char * resp, size_t resp_size));
NOVA_API int nova_server_is_running(struct nova_server * server);

// ============================================================================
// NOVA CLI API
// ============================================================================

NOVA_API int nova_cli_main(int argc, char ** argv);
NOVA_API int nova_server_main(int argc, char ** argv);
NOVA_API int nova_download_main(int argc, char ** argv);
NOVA_API int nova_convert_main(int argc, char ** argv);
NOVA_API int nova_quantize_main(int argc, char ** argv);
NOVA_API int nova_perplexity_main(int argc, char ** argv);
NOVA_API int nova_benchmark_main(int argc, char ** argv);
NOVA_API int nova_train_main(int argc, char ** argv);
NOVA_API int nova_finetune_main(int argc, char ** argv);
NOVA_API int nova_eval_main(int argc, char ** argv);
NOVA_API int nova_profile_main(int argc, char ** argv);

// ============================================================================
// NOVA UTILITY API
// ============================================================================

NOVA_API const char * nova_ftype_name(enum nova_ftype ftype);
NOVA_API const char * nova_flash_attn_type_name(enum nova_flash_attn_type flash_attn_type);
NOVA_API const char * nova_build_info(void);
NOVA_API void nova_print_version(void);
NOVA_API int nova_parse_bool(const char * str);
NOVA_API float nova_parse_float(const char * str);
NOVA_API int nova_parse_int(const char * str);
NOVA_API void nova_seed_random(uint32_t seed);
NOVA_API uint32_t nova_random_u32(void);
NOVA_API float nova_random_float(void);
NOVA_API void nova_sleep_ms(int ms);
NOVA_API int nova_file_exists(const char * path);
NOVA_API size_t nova_file_size(const char * path);
NOVA_API char * nova_file_read(const char * path);
NOVA_API int nova_file_write(const char * path, const char * data, size_t size);
NOVA_API char * nova_dirname(const char * path);
NOVA_API char * nova_basename(const char * path);
NOVA_API int nova_mkdir_p(const char * path);
NOVA_API int nova_rm_rf(const char * path);
NOVA_API char * nova_strdup(const char * str);
NOVA_API char * nova_str_cat(const char * a, const char * b);
NOVA_API char * nova_str_replace(const char * str, const char * old, const char * new);
NOVA_API int nova_str_startswith(const char * str, const char * prefix);
NOVA_API int nova_str_endswith(const char * str, const char * suffix);
NOVA_API char * nova_trim(char * str);
NOVA_API char * nova_trim_left(char * str);
NOVA_API char * nova_trim_right(char * str);
NOVA_API char * nova_to_lower(const char * str);
NOVA_API char * nova_to_upper(const char * str);

// ============================================================================
// NOVA DEBUG API
// ============================================================================

NOVA_API void nova_debug_dump_tensor(struct nova_tensor * t);
NOVA_API void nova_debug_dump_model(struct nova_model * model);
NOVA_API void nova_debug_dump_memory(struct nova_context * ctx);
NOVA_API void nova_debug_dump_cache(struct nova_kv_cache * cache);
NOVA_API void nova_debug_dump_graph(void * gf);
NOVA_API void nova_debug_enable(int enable);
NOVA_API int nova_debug_is_enabled(void);
NOVA_API void nova_debug_set_level(int level);
NOVA_API int nova_debug_get_level(void);

// ============================================================================
// NOVA PROFILE API
// ============================================================================

NOVA_API void nova_profile_start(void);
NOVA_API void nova_profile_stop(void);
NOVA_API void nova_profile_reset(void);
NOVA_API void nova_profile_dump(const char * path);
NOVA_API void nova_profile_enable_gpu(int enable);
NOVA_API void nova_profile_enable_cpu(int enable);
NOVA_API void nova_profile_enable_memory(int enable);

// ============================================================================
// NOVA HARDWARE API
// ============================================================================

NOVA_API int nova_hardware_init(void);
NOVA_API void nova_hardware_shutdown(void);
NOVA_API int nova_hardware_get_gpu_count(void);
NOVA_API int nova_hardware_get_gpu_memory(int gpu_id, size_t * total, size_t * free);
NOVA_API int nova_hardware_get_gpu_utilization(int gpu_id, float * util);
NOVA_API int nova_hardware_get_gpu_temp(int gpu_id, float * temp);
NOVA_API int nova_hardware_get_gpu_power(int gpu_id, float * power);
NOVA_API int nova_hardware_select_device(int gpu_id);
NOVA_API int nova_hardware_get_current_device(void);
NOVA_API void nova_hardware_sync(void);

// ============================================================================
// END
// ============================================================================

#ifdef __cplusplus
}
#endif

#endif // NOVA_H
