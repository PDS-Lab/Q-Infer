#pragma once

#include <stdint.h>
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif

#define STAT_MAX_LAYER_NUM 100
#define STAT_MAX_FFN_NUERON_NUM 32768

struct Statistic;
struct global_stat_t {
  struct Statistic *op_stat;
  struct Statistic *op_stat_gpu;
  struct Statistic *mem_stat;
  struct Statistic *other_stat;
  struct Statistic *layer_stat[STAT_MAX_LAYER_NUM];
  bool inited;
};

extern struct global_stat_t global_stat;

void stat_init(void);
void stat_destory(void);
bool stat_empty(struct Statistic *stat);
void stat_record(struct Statistic *, const char *name, int64_t delta, enum ggml_backend_type bt, enum ggml_backend_type exect);
void stat_show(struct Statistic *);
void stat_reset(struct Statistic *);


struct ratio_hist {
  // runtime stat
  int hist[STAT_MAX_LAYER_NUM][21];
  float ratio[STAT_MAX_LAYER_NUM];
  bool need_hist;
  int cnt;  // 单独的计数器
  // result
  float avg_ratio;
};

struct sparsity_stat {
  struct ratio_hist active;       // 卸载到GPU上的神经元被使用的比例，越大说明命中率越高

  struct ratio_hist precision;    // 卸载到GPU上的神经元，参与了当次多少的计算量，越大越好
  struct ratio_hist offload_gain; // GPU上神经元参与的计算量比例 与 神经元卸载量之间比例 之间的 比值，越大说明算法越高效，集中性越强

  struct ratio_hist recall;       // 卸载到GPU上的神经元，召回了多少激活神经元，越大越好
  struct ratio_hist recall_gain;  // 召回率与卸载率之间的比例，越大越好

  struct ratio_hist sparsity;     // 联合上下文稀疏性
};

void init_sparsity_stat();
void init_hist(struct ratio_hist *hist);
void summarize_hist(struct ratio_hist *hist);
void show_hist(struct ratio_hist *hist);
void show_sparsity();


struct spatiotemporal_locality {
  struct ratio_hist layer_locality;
  struct ratio_hist token_locality;
  float *sparse_idx[STAT_MAX_LAYER_NUM];
  float *pred_idx[STAT_MAX_LAYER_NUM];
};

void init_spatiotemporal_stat(void);
void destory_spatiotemporal_stat(void);
void show_locality(void);

extern struct spatiotemporal_locality g_st_locality;
extern struct sparsity_stat g_sparsity;
extern struct ggml_tensor *g_gpu_indx[STAT_MAX_LAYER_NUM];
extern int net_layer_num;             // 网络总共的层次
extern int g_cycle;                   // 总共遍历整个网络多少轮
extern int g_batch_size;              // 推理batch size
extern int g_ffn_neuron_num;          // ffn层神经元个数
extern bool g_token_locality_flag;    // 当单个请求结束或开始时，用来控制是否该计算token局部性

extern size_t cpu_start;
extern size_t cpu_stop;

struct kvcache_split {
  bool flag;
  float ratio[STAT_MAX_LAYER_NUM];
  float update_step;
};

float update_ratio(float cur, int64_t cpu_time, int64_t gpu_time);
void show_kvcache_ratio();

void add_ratio_curve(float ra);
void show_ratio_curve();

extern struct kvcache_split g_kvc_split;

struct neuron_freq {
  uint64_t history_freq[STAT_MAX_FFN_NUERON_NUM];
  uint64_t last_freq[STAT_MAX_FFN_NUERON_NUM];
  uint64_t last_window[STAT_MAX_FFN_NUERON_NUM][10];
  int pos;
};

struct neuron_cache_meta {
  bool neuron_map[STAT_MAX_FFN_NUERON_NUM];
  uint64_t pos_map[STAT_MAX_FFN_NUERON_NUM];
  size_t cache_size;
};

struct neuron_cache_simu {
  struct neuron_cache_meta meta[STAT_MAX_LAYER_NUM];
  struct neuron_freq freq[STAT_MAX_LAYER_NUM];
  int layer_k[STAT_MAX_LAYER_NUM];
  int layer;
};

extern struct neuron_cache_simu g_neuron_cache;

double freq_score(int64_t hist_freq, int64_t last_freq);

void show_cache_simu();
#ifdef __cplusplus
}
#endif
