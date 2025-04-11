#include "statistic.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <threads.h>
#include <vector>

static inline bool(likely)(bool x) { return __builtin_expect((x), true); }
static inline bool(unlikely)(bool x) { return __builtin_expect((x), false); }

/**
 * @file timer.h
 * @brief Helper functions for timers
 */
#include <stdexcept>
#include <stdint.h>
#include <string>

struct StatData {
  int cnt{};
  uint64_t sum{};
  uint64_t max_val{};
  uint64_t min_val{UINT64_MAX};
  uint64_t avg_val{};
  ggml_backend_type op_t;
  ggml_backend_type exec_t;

  void add(uint64_t delta) {
    cnt++;
    sum += delta;
    max_val = std::max(delta, max_val);
    min_val = std::min(delta, min_val);
    avg_val = sum / cnt;
  }
};

struct Statistic {
  std::map<std::string, StatData> stat_impl;
  std::vector<std::string> keys;

  void record(const std::string &name, uint64_t delta, ggml_backend_type bt, ggml_backend_type exect) {
    auto it = stat_impl.find(name);
    if (unlikely(it == stat_impl.end())) {
      keys.push_back(name);
      stat_impl[name].add(delta);
      stat_impl[name].op_t = bt;
      stat_impl[name].exec_t = exect;
    } else {
      it->second.add(delta);
    }
  }

  void show() {
    printf("\n");
    for (auto &k : keys) {
      show(k);
    }
  }


  const char *backend_map[11] = {
    "CPU", "", "", "", "", "", "", "", "", "",
    "GPU"
  };

  void show(const std::string &name) {
    auto it = stat_impl.find(name);
    if (unlikely(it == stat_impl.end())) {
      throw std::runtime_error("no that statistic item:" + name);
    }
    auto &item = it->second;
    printf("%-40s P:%s C:%s [Total count:%10d]   [Total time used:%10luus]   "
           "[Average:%10luus]   [Max:%10luus]   [Min:%10luus]\n",
           name.c_str(), backend_map[item.op_t], backend_map[item.exec_t], item.cnt, item.sum, item.avg_val,
           item.max_val, item.min_val);
  }
};

struct global_stat_t global_stat = {
    nullptr, nullptr, nullptr, nullptr, {nullptr}, false,
};

void stat_record(struct Statistic *stat, const char *name, int64_t delta,
                 ggml_backend_type bt, ggml_backend_type exect) {
  stat->record(name, delta, bt, exect);
}
void stat_show(struct Statistic *stat) { stat->show(); }
void stat_reset(struct Statistic *stat) {
  stat->stat_impl.clear();
  stat->keys.clear();
}
bool stat_empty(struct Statistic *stat) { return stat->stat_impl.empty(); }

void stat_init() {
  if (global_stat.inited) {
    return;
  }
  global_stat.op_stat = new Statistic();
  global_stat.op_stat_gpu = new Statistic();
  global_stat.mem_stat = new Statistic();
  global_stat.other_stat = new Statistic();
  for (int i = 0; i < STAT_MAX_LAYER_NUM; i++) {
    global_stat.layer_stat[i] = new Statistic();
  }
}

void stat_destory() {
  delete global_stat.op_stat;
  delete global_stat.op_stat_gpu;
  delete global_stat.mem_stat;
  delete global_stat.other_stat;
  for (int i = 0; i < STAT_MAX_LAYER_NUM; i++) {
    delete global_stat.layer_stat[i];
  }
}

void init_hist(struct ratio_hist *hist) {
  for (int i = 0; i < STAT_MAX_LAYER_NUM; i++) {
    hist->ratio[i] = 0;
    memset(hist->hist[i], 0, sizeof(hist->hist[i]));
  }
  hist->cnt = 0;
  hist->avg_ratio = 0;
  hist->need_hist = false;
}

void init_sparsity_stat() {
  init_hist(&g_sparsity.sparsity);
  init_hist(&g_sparsity.active);
  init_hist(&g_sparsity.precision);
  init_hist(&g_sparsity.recall);
}

struct spatiotemporal_locality g_st_locality;

void init_spatiotemporal_stat() {
  init_hist(&g_st_locality.layer_locality);  
  init_hist(&g_st_locality.token_locality);
  for (int i = 0; i < STAT_MAX_LAYER_NUM; i++) {
    g_st_locality.sparse_idx[i] = (float *)malloc(sizeof(float) * g_ffn_neuron_num);
    g_st_locality.pred_idx[i] = (float *)malloc(sizeof(float) * g_ffn_neuron_num);
  } 
}

void destory_spatiotemporal_stat() {
  for (int i = 0; i < STAT_MAX_LAYER_NUM; i++) {
    free(g_st_locality.sparse_idx[i]);
    free(g_st_locality.pred_idx[i]);
  }
}

void show_locality() {
  printf("\nInter-Layer locality :\n");
  g_st_locality.layer_locality.need_hist = true;
  summarize_hist(&g_st_locality.layer_locality);
  show_hist(&g_st_locality.layer_locality);

  printf("\nInter-Token locality :\n");
  g_st_locality.token_locality.need_hist = true;
  summarize_hist(&g_st_locality.token_locality);
  show_hist(&g_st_locality.token_locality);
}

struct sparsity_stat g_sparsity = {};

void summarize_hist(struct ratio_hist *hist) {
  float sum = 0;
  int cycle = hist->cnt != 0 ? hist->cnt : g_cycle;
  for (int i = 0; i < net_layer_num; i++) {
    hist->ratio[i] /= cycle;
    sum += hist->ratio[i];
  }
  hist->avg_ratio = sum / net_layer_num;
}

void show_hist(struct ratio_hist *hist) {
  printf("rounds %d\n", g_cycle);
  printf("average ratio : %f\n", hist->avg_ratio);
  printf("level ratio:\n");
  for (int i = 0; i < net_layer_num; i++) {
    printf("[level %d ratio: %f]\n", i, hist->ratio[i]);
  }
  int cycle = hist->cnt != 0 ? hist->cnt : g_cycle;
  if (hist->need_hist) {
    printf("\n<============ historgram ===========>:\n");
    for (int i = 0; i < net_layer_num; i++) {
      printf("level %d hit ratio hist:", i);
      for (int j = 0; j < 20; j++) {
        printf("[%d-%d]:%f  ", j*5, (j+1)*5, hist->hist[i][j] * 1.0 / cycle);
      }
      printf("[100]:%f  ", hist->hist[i][20] * 1.0 / cycle);
      printf("\n");
    }
  }
}

void show_sparsity() {
  printf("\nContexual Sparsity:\n");
  g_sparsity.sparsity.need_hist = true;
  summarize_hist(&g_sparsity.sparsity);
  show_hist(&g_sparsity.sparsity);

  printf("\nActive:\n");
  g_sparsity.active.need_hist = true;
  summarize_hist(&g_sparsity.active);
  show_hist(&g_sparsity.active);

  printf("\nPrecision:\n");
  g_sparsity.precision.need_hist = true;
  summarize_hist(&g_sparsity.precision);
  show_hist(&g_sparsity.precision);

  printf("\nPrecision Gain:\n");
  summarize_hist(&g_sparsity.offload_gain);
  show_hist(&g_sparsity.offload_gain);

  printf("\nRecall:\n");
  g_sparsity.recall.need_hist = true;
  summarize_hist(&g_sparsity.recall);
  show_hist(&g_sparsity.recall);

  printf("\nRecall Gain:\n");
  summarize_hist(&g_sparsity.recall_gain);
  show_hist(&g_sparsity.recall_gain);
}

void show_cache_simu() {
  printf("\nRecall:\n");
  g_sparsity.recall.need_hist = true;
  summarize_hist(&g_sparsity.recall);
  show_hist(&g_sparsity.recall);

  printf("\nRecall Gain:\n");
  summarize_hist(&g_sparsity.recall_gain);
  show_hist(&g_sparsity.recall_gain);  
}

ggml_tensor *g_gpu_indx[STAT_MAX_LAYER_NUM] = {NULL};
int net_layer_num = 0;
int g_cycle = 0;
int g_batch_size = 0;
int g_ffn_neuron_num = 0 ;  // ffn层神经元个数
bool g_token_locality_flag = false;

size_t cpu_start = 0;
size_t cpu_stop = 0;

struct kvcache_split g_kvc_split;

float update_ratio(float cur, int64_t cpu_time, int64_t gpu_time) {
  if (!g_kvc_split.flag) {
    return cur;
  }
  int64_t delta_time = gpu_time - cpu_time;
  if (std::abs(delta_time) < (cpu_time + gpu_time) * 0.025) {
    // 差距太小就不调了
    return cur;
  }

  // float update = (cur * delta_time * (1 - cur)) / (delta_time * cur + cpu_time) / 2;
  float update = g_kvc_split.update_step;
  if (delta_time < 0) {
    update = -update;
  }
  assert(update + cur > 0 && update + cur < 1);
  return update + cur;
}

void show_kvcache_ratio() {
  printf("level ratio:\n");
  for (int i = 0; i < net_layer_num; i++) {
    printf("[level %d ratio: %f]\n", i, g_kvc_split.ratio[i]);
  }
}

std::vector<float> bu;

void add_ratio_curve(float ra) {
  bu.push_back(ra);
}

void show_ratio_curve() {
  printf("ratio curve:\n");
  for (auto ratio : bu) {
    printf("ratio: %f]\n", ratio);
  }
}

struct neuron_cache_simu g_neuron_cache;

double freq_score(int64_t hist_freq, int64_t last_freq) {
  return (double)hist_freq * 0.1 + (double)last_freq * 0.9;
}