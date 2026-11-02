/* qmesh_moe_cpp: the GX cold-expert forward with no Python on the request path (see the header).
 *
 * The sequence here is qmesh/exl3_moe_cuda.CudaExl3Moe.forward, step for step, with three
 * differences that are the point of the file:
 *
 *  - the request's x/ids/w are copied straight out of the pinned slab into persistent device
 *    buffers (cudaMemcpyAsync), and the answer is copied straight back into the pinned response
 *    slot. No numpy view, no .copy(), no .to(), no .cpu().numpy(), no tensor per element.
 *  - every buffer the kernels write is allocated once and reused (monotone grow-on-demand), so a
 *    served call runs no allocator code at all, and the bf16->half / weight conversions that
 *    `.half()` and `.float()` did are two kernels.
 *  - the alignment (align_block) is two kernels plus one small pinned count vector instead of
 *    sort/bincount/cumsum/repeat_interleave with two host read-backs.
 *
 * The math is the vendored cuda-exl3 kernels, called through the torch dispatcher exactly as the
 * Python path called them (`cuda_exl3_C::exl3_moe_*`), so a change in the vendor extension reaches
 * both paths at once. Numerics differ from the Python path only in the order of the
 * down-projection's fp32 atomics within an expert's block (measured in the ticket).
 *
 * The piece ladder is duplicated from worker_seam_server._pick_step, with the same constants and
 * the same SEAM_WS_* environment: the C hook is installed *instead of* the Python _forward, so it
 * has to make the same piece decision or the two paths would not be comparable. Keep them in
 * sync; the SEAM_WS_PICK line it prints is the same one, so a run that picks differently from the
 * Python path is visible in the log.
 */
#include "qmesh_moe_cpp.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int QMC_NL = 40;
constexpr int QMC_MAX_ROWS = 65536;
constexpr int QMC_MAX_EXPERTS = 384; /* the model's 384 routed experts: the per-group stride of the
                                      * count/offset vectors, and the padding bound of an alignment */
constexpr int kBlockMTiers[4] = {16, 32, 64, 128}; /* exl3_moe_cuda.BLOCK_M_TIERS */
constexpr int kBlockMRows = 24;                   /* exl3_moe_cuda.BLOCK_M_ROWS */
constexpr int kRefuseMem = 4;
constexpr int kRefuseCompute = 5;

thread_local std::string g_err;

void set_err(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void set_err(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_err = buf;
}

int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  if (!v || !*v) return fallback;
  return atoi(v);
}

/* -- /proc readings the piece picker and the memory refusal share with worker_seam_server -------- */
long meminfo_kib(const char* key) {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return -1;
  char line[256];
  const size_t klen = strlen(key);
  long out = -1;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, key, klen) == 0 && line[klen] == ':') {
      out = atol(line + klen + 1);
      break;
    }
  }
  fclose(f);
  return out;
}

int avail_mib() {
  const long kib = meminfo_kib("MemAvailable");
  return kib < 0 ? -1 : (int)(kib / 1024);
}

int swap_used_mib() {
  const long tot = meminfo_kib("SwapTotal"), fre = meminfo_kib("SwapFree");
  if (tot < 0 || fre < 0) return -1;
  return (int)((tot - fre) / 1024);
}

/* -- dispatcher ---------------------------------------------------------------------------------
 * Boxed calls keep this file independent of the extension's generated headers: the schema is the
 * one cuda_exl3 registered, and a signature drift fails loudly here instead of silently. */
c10::OperatorHandle op_handle(const char* name) {
  static std::mutex m;
  static std::unordered_map<std::string, c10::OperatorHandle> cache;
  std::lock_guard<std::mutex> lk(m);
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;
  auto h = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  cache.emplace(name, h);
  return h;
}

at::Tensor dev_tensor(void* p, at::IntArrayRef sizes, at::ScalarType dt, int dev) {
  return at::from_blob(p, sizes, at::TensorOptions().dtype(dt).device(at::kCUDA, dev));
}

c10::IValue iv_opt(const at::Tensor& t, bool has) { return has ? c10::IValue(t) : c10::IValue(); }
c10::IValue iv_i64(int64_t v) { return c10::IValue(v); }

void moe_had_in(const at::Tensor& x, const at::Tensor& out, const at::Tensor& suh,
                const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                const at::Tensor& n_rows, int block_m, int top_k, int m_valid) {
  std::vector<c10::IValue> a{c10::IValue(x), c10::IValue(out), c10::IValue(suh),
                             c10::IValue(sorted_ids), c10::IValue(expert_ids),
                             c10::IValue(n_rows), iv_i64(block_m), iv_i64(top_k), iv_i64(m_valid),
                             c10::IValue(true)};
  op_handle("cuda_exl3_C::exl3_moe_had_in").callBoxed(a);
}

void moe_glu(const at::Tensor& x, const at::Tensor& out, const at::Tensor& suh,
             const at::Tensor& expert_ids, const at::Tensor& n_rows, int block_m) {
  std::vector<c10::IValue> a{c10::IValue(x), c10::IValue(out), c10::IValue(suh),
                             c10::IValue(expert_ids), c10::IValue(n_rows), iv_i64(block_m)};
  op_handle("cuda_exl3_C::exl3_moe_glu_had_in").callBoxed(a);
}

at::Tensor moe_gemm(const at::Tensor& a_had, const at::Tensor& trellis, const at::Tensor& suh,
                    const at::Tensor& svh, const at::Tensor& expert_ids, const at::Tensor& n_rows,
                    const std::vector<int64_t>& group_n, int cb, int block_m, at::ScalarType dt,
                    const at::Tensor& sorted_ids, bool have_sids, const at::Tensor& topk_w,
                    bool have_w, int num_tokens, int top_k, const at::Tensor& out, bool have_out,
                    bool accumulate) {
  std::vector<c10::IValue> a{
      c10::IValue(a_had),
      c10::IValue(trellis),
      c10::IValue(suh),
      c10::IValue(svh),
      c10::IValue(expert_ids),
      c10::IValue(n_rows),
      c10::IValue(c10::List<int64_t>(group_n)),
      iv_i64(cb),
      iv_i64(block_m),
      c10::IValue(dt),
      iv_opt(sorted_ids, have_sids),
      iv_opt(topk_w, have_w),
      iv_i64(num_tokens),
      iv_i64(top_k),
      iv_opt(out, have_out),
      c10::IValue(accumulate)};
  // Boxed convention: the kernel drops its inputs off the stack and pushes its outputs, so after
  // the call the stack holds exactly the single returned tensor.
  op_handle("cuda_exl3_C::exl3_moe_gemm").callBoxed(a);
  return a.at(0).toTensor();
}

/* exl3_moe_cuda.pick_block_m, including CUDA_EXL3_MOE_BLOCK_M: first minimum wins, as min(..., key=) */
int pick_block_m(int live, int n_experts) {
  static const int override_bm = env_int("CUDA_EXL3_MOE_BLOCK_M", 0);
  const double per = (double)live / (n_experts > 0 ? n_experts : 1);
  int best = kBlockMTiers[0];
  double best_cost = 1e300;
  for (int b : kBlockMTiers) {
    const double q = std::ceil(per / b);
    const double cost = q * b + kBlockMRows * q;
    if (cost < best_cost) {
      best_cost = cost;
      best = b;
    }
  }
  if (override_bm) best = override_bm;
  return best;
}

struct Group {
  qmc_group g{};
  at::Tensor t13, suh13, svh13, t2, suh2, svh2; /* registered once, alive for the process */
  int last_live = 0; /* live picks of the previous call: block_m for this one, no sync needed */
};

struct Layer {
  bool have = false;
  int n_groups = 0;
  Group grp[QMC_MAX_GROUPS];
  /* The alignment of the last piece served for this layer, for qmc_dump_align: only meaningful
   * right after a call, and only filled when the dump is on. */
  int dump_rows = 0;
  int dump_bm[QMC_MAX_GROUPS]{}, dump_nblk[QMC_MAX_GROUPS]{}, dump_total[QMC_MAX_GROUPS]{};
};

/* Every device buffer a served call touches, allocated once and grown on demand. */
struct Buffers {
  int cap_rows = 0;
  int cap_sorted = 0;
  int cap_pad = 0; /* rows of the a13/inter/a2 trio currently allocated */
  int n_groups = 0;
  int n_cold_max = 0;

  uint16_t* x_bf = nullptr;
  void* x_h = nullptr;
  int32_t* ids = nullptr;
  uint16_t* w_bf = nullptr;
  float* w_f = nullptr;

  int32_t* counts = nullptr;   /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* cursor = nullptr;   /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* blk_off = nullptr;  /* [n_groups * QMC_MAX_EXPERTS] */
  int32_t* bad = nullptr;      /* [1] */
  int32_t* n_rows = nullptr;   /* [n_groups] */
  int32_t* nblk = nullptr;     /* [n_groups]: blocks per group, written by qmc_k_align */
  int32_t* live = nullptr;     /* [n_groups]: live picks per group, written by qmc_k_align */
  int32_t* sorted = nullptr;   /* [cap_sorted] */
  int32_t* eids = nullptr;     /* [n_groups * cap_sorted]: per-block expert id, one slot per group */

  void* a13 = nullptr;   /* half [2, cap_pad, H] */
  void* inter = nullptr; /* half [cap_pad, 2I] */
  void* a2 = nullptr;    /* half [1, cap_pad, I] */
  float* acc = nullptr; /* float [cap_rows, H] */
  void* out16 = nullptr; /* half/bf16 [cap_rows, H]: the answer when the slot carries halves */

  int32_t* h_plan = nullptr; /* pinned [2 * n_groups]: nblk then live per group, one readback */
  int32_t* h_bad = nullptr;  /* pinned [1] */
  int32_t* h_sorted = nullptr; /* pinned [n_groups * cap_sorted], only with SEAM_MOE_CPP_DUMP */
  int32_t* h_eids = nullptr;   /* pinned [n_groups * cap_sorted], only with SEAM_MOE_CPP_DUMP */

  ~Buffers() {
    auto cf = [](void*& p) {
      if (p) {
        cudaFree(p);
        p = nullptr;
      }
    };
    cf((void*&)x_bf);
    cf(x_h);
    cf((void*&)ids);
    cf((void*&)w_bf);
    cf((void*&)w_f);
    cf((void*&)counts);
    cf((void*&)cursor);
    cf((void*&)blk_off);
    cf((void*&)bad);
    cf((void*&)n_rows);
    cf((void*&)nblk);
    cf((void*&)live);
    cf((void*&)eids);
    cf(a13);
    cf(inter);
    cf(a2);
    cf((void*&)acc);
    cf(out16);
    auto ch = [](void*& p) {
      if (p) {
        cudaFreeHost(p);
        p = nullptr;
      }
    };
    ch((void*&)h_plan);
    ch((void*&)h_bad);
    ch((void*&)h_sorted);
    ch((void*&)h_eids);
  }
};

struct Ctx {
  int dev = 0;
  int hidden = 0;
  int topk = 0;
  int ws_rows = 0;
  bool ws_auto = true;
  int ws_guard = 96;
  int ws_hyst = 128;
  int swap_warn = 512;
  int fwd_floor = 0;
  int out_mode = 0; /* 0 = fp32 response slot, 1 = bf16, 2 = fp16 (SEAM_MOE_CPP_OUT) */
  int block_m = 0;  /* 0 = pick_block_m from the previous call's live count (see run_piece) */
  bool dump = false; /* SEAM_MOE_CPP_DUMP: mirror each group's alignment into host buffers */
  bool trace = false;
  int ws_pick = -1;
  int n_groups_max = 0;
  std::mutex mu;
  Layer layers[QMC_NL];
  Buffers b;
  int64_t calls = 0;
  int64_t refused = 0;
  int64_t last_wall = 0;
  int64_t last_dev = 0;
  bool reported = false;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
};

bool cuda_alive() {
  /* A dead context fails this and keeps failing; a refusal for memory does not. Same test as
   * worker_seam_server._cuda_ok, from the C side. */
  void* p = nullptr;
  if (cudaMalloc(&p, 4) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const cudaError_t e = cudaMemset(p, 0, 4);
  cudaFree(p);
  cudaGetLastError();
  return e == cudaSuccess;
}

int pick_step(Ctx& c, int rows) {
  if (c.ws_rows <= 0) return rows;
  if (!c.ws_auto) return std::min(c.ws_rows, rows);
  const int avail = avail_mib();
  const int swapped = swap_used_mib();
  const bool paging = swapped > c.swap_warn;
  const int ladder[2] = {2048, 1024}; /* worker_seam_server.WS_LADDER */
  const int kib_per_row = 235;        /* worker_seam_server.WS_KIB_PER_ROW */
  int pick = -1;
  if (!paging) {
    for (int n : ladder) {
      if (n > c.ws_rows) continue;
      const int slack = (c.ws_pick < 0 || n <= c.ws_pick) ? 0 : c.ws_hyst;
      /* ws_mib_for(rows, n): a piece larger than the request costs only the request. */
      const double need = (double)std::min(n, rows) * kib_per_row / 1024.0;
      if (avail < 0 || need + c.ws_guard + slack <= avail) {
        pick = n;
        break;
      }
    }
  }
  if (pick < 0) pick = std::min(c.ws_rows, ladder[1]);
  if (pick != c.ws_pick) {
    c.ws_pick = pick;
    fprintf(stderr,
            "SEAM_WS_PICK rows=%d ws=%d avail_mib=%d swapped_mib=%d need_mib=%d pieces=%d\n", rows,
            pick, avail, swapped,
            (int)((double)std::min(pick, rows) * kib_per_row / 1024.0 + c.ws_guard),
            (rows + pick - 1) / pick);
    fflush(stderr);
  }
  return std::min(pick, rows);
}

/* -- buffers ------------------------------------------------------------------------------------ */
bool ensure_buffers(Ctx& c, int rows, int total_pad) {
  Buffers& b = c.b;
  const int H = c.hidden, T = c.topk;
  auto grow = [](void** p, size_t bytes) {
    if (*p) {
      cudaFree(*p);
      *p = nullptr;
    }
    return cudaMalloc(p, bytes);
  };
  if (rows > b.cap_rows) {
    if (grow((void**)&b.x_bf, (size_t)rows * H * 2) != cudaSuccess) return false;
    if (grow(&b.x_h, (size_t)rows * H * 2) != cudaSuccess) return false;
    if (grow((void**)&b.ids, (size_t)rows * T * 4) != cudaSuccess) return false;
    if (grow((void**)&b.w_bf, (size_t)rows * T * 2) != cudaSuccess) return false;
    if (grow((void**)&b.w_f, (size_t)rows * T * 4) != cudaSuccess) return false;
    if (grow((void**)&b.acc, (size_t)rows * H * 4) != cudaSuccess) return false;
    if (grow(&b.out16, (size_t)rows * H * 2) != cudaSuccess) return false;
    b.cap_rows = rows;
  }
  /* Sized for the largest group count of any registered layer: layers may disagree (a pack whose
   * bitrates do not cover every layer), and a per-layer flip would reallocate on every call. */
  const int n_groups = c.n_groups_max;
  if (n_groups > b.n_groups) {
    if (grow((void**)&b.counts, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.cursor, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.blk_off, (size_t)n_groups * QMC_MAX_EXPERTS * 4) != cudaSuccess)
      return false;
    if (grow((void**)&b.n_rows, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (grow((void**)&b.nblk, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (grow((void**)&b.live, (size_t)n_groups * 4) != cudaSuccess) return false;
    if (b.h_plan) cudaFreeHost(b.h_plan);
    b.h_plan = nullptr;
    if (cudaHostAlloc((void**)&b.h_plan, (size_t)2 * n_groups * 4, cudaHostAllocPortable) !=
        cudaSuccess)
      return false;
    b.n_groups = n_groups;
  }
  if (!b.bad) {
    if (grow((void**)&b.bad, 4) != cudaSuccess) return false;
    if (cudaHostAlloc((void**)&b.h_bad, 4, cudaHostAllocPortable) != cudaSuccess) return false;
  }
  /* Blocks are one per ceil(count/block_m), so an alignment is bounded by the picks plus one
   * padding block per expert, per group: cap >= total of every group. block_m is not known here
   * yet, so the bound takes the slowest tier; the trio itself is then grown to the exact total the
   * alignment reports. */
  const int cap = total_pad + (QMC_MAX_EXPERTS + 1) * kBlockMTiers[3];
  if (cap > b.cap_sorted) {
    if (grow((void**)&b.sorted, (size_t)cap * 4) != cudaSuccess) return false;
    if (grow((void**)&b.eids, (size_t)n_groups * cap * 4) != cudaSuccess) return false;
    if (b.h_sorted) cudaFreeHost(b.h_sorted);
    if (b.h_eids) cudaFreeHost(b.h_eids);
    b.h_sorted = b.h_eids = nullptr;
    if (c.dump) {
      if (cudaHostAlloc((void**)&b.h_sorted, (size_t)n_groups * cap * 4,
                        cudaHostAllocPortable) != cudaSuccess)
        return false;
      if (cudaHostAlloc((void**)&b.h_eids, (size_t)n_groups * cap * 4, cudaHostAllocPortable) !=
          cudaSuccess)
        return false;
    }
    b.cap_sorted = cap;
  }
  if (c.dump && !b.h_sorted) {
    if (cudaHostAlloc((void**)&b.h_sorted, (size_t)n_groups * b.cap_sorted * 4,
                      cudaHostAllocPortable) != cudaSuccess)
      return false;
    if (cudaHostAlloc((void**)&b.h_eids, (size_t)n_groups * b.cap_sorted * 4,
                      cudaHostAllocPortable) != cudaSuccess)
      return false;
  }
  return true;
}

bool grow_trio(Ctx& c, int total_pad, int inter) {
  Buffers& b = c.b;
  if (total_pad <= b.cap_pad) return true;
  auto grow = [](void** p, size_t bytes) {
    if (*p) {
      cudaFree(*p);
      *p = nullptr;
    }
    return cudaMalloc(p, bytes);
  };
  const size_t H = (size_t)c.hidden;
  if (grow(&b.a13, (size_t)2 * total_pad * H * 2) != cudaSuccess) return false;
  if (grow(&b.inter, (size_t)total_pad * 2 * (size_t)inter * 2) != cudaSuccess) return false;
  if (grow(&b.a2, (size_t)total_pad * (size_t)inter * 2) != cudaSuccess) return false;
  b.cap_pad = total_pad;
  return true;
}

/* -- one piece ---------------------------------------------------------------------------------- */
int run_piece(Ctx& c, int layer, int rows, const uint8_t* xp, const int32_t* idp,
              const uint8_t* wp, float* outp) {
  Layer& L = c.layers[layer];
  const int H = c.hidden, T = c.topk;
  const int m_valid = rows * T;
  cudaStream_t s = at::cuda::getCurrentCUDAStream(c.dev).stream();

  if (!ensure_buffers(c, rows, m_valid)) {
    set_err("buffer alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseMem;
  }
  Buffers& b = c.b;

  /* Request slot -> device. These are the copies worker_seam_server did with numpy .copy() and
   * .to(device); the bf16->half and the routing-weight conversions are the two kernels after. */
  if (cudaMemcpyAsync(b.x_bf, xp, (size_t)rows * H * 2, cudaMemcpyHostToDevice, s) != cudaSuccess ||
      cudaMemcpyAsync(b.ids, idp, (size_t)rows * T * 4, cudaMemcpyHostToDevice, s) != cudaSuccess ||
      cudaMemcpyAsync(b.w_bf, wp, (size_t)rows * T * 2, cudaMemcpyHostToDevice, s) != cudaSuccess) {
    set_err("H2D from the pinned slab failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  qmc_k_bf16_to_half(b.x_bf, b.x_h, rows * H, s);
  qmc_k_weights(b.ids, b.w_bf, b.w_f, rows * T, s);

  /* Histogram and alignment, both on device. block_m is a function of the live picks per expert;
   * the live count of the previous call stands in for this one (a request moves it by a few
   * percent, and the tier is coarse), or the picks bound m_valid answers the first call of a
   * bucket. Nothing here reads the counts back, so the per-group arithmetic align_block did on the
   * host - offsets, repeat_interleave, the padded row count - is all in qmc_k_align, and the only
   * readback before the gemms is 8 bytes per group. */
  if (cudaMemsetAsync(b.bad, 0, 4, s) != cudaSuccess ||
      cudaMemsetAsync(b.counts, 0, (size_t)L.n_groups * QMC_MAX_EXPERTS * 4, s) != cudaSuccess) {
    set_err("histogram clear failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  for (int g = 0; g < L.n_groups; g++) {
    const qmc_group& q = L.grp[g].g;
    qmc_k_hist(b.ids, m_valid, q.n_cold, q.remap, q.n_experts,
               b.counts + (size_t)g * QMC_MAX_EXPERTS, b.bad, s);
  }
  int bm[QMC_MAX_GROUPS] = {0};
  for (int g = 0; g < L.n_groups; g++) {
    const qmc_group& q = L.grp[g].g;
    const int hint = L.grp[g].last_live > 0 ? L.grp[g].last_live : m_valid;
    bm[g] = c.block_m > 0 ? c.block_m : pick_block_m(hint, q.n_experts);
    qmc_k_align(b.counts + (size_t)g * QMC_MAX_EXPERTS, q.n_experts, bm[g],
                b.blk_off + (size_t)g * QMC_MAX_EXPERTS,
                b.eids + (size_t)g * b.cap_sorted, b.nblk + g, b.live + g, s);
  }
  if (cudaMemcpyAsync(b.h_plan, b.nblk, (size_t)L.n_groups * 4, cudaMemcpyDeviceToHost, s) !=
          cudaSuccess ||
      cudaMemcpyAsync(b.h_plan + L.n_groups, b.live, (size_t)L.n_groups * 4,
                      cudaMemcpyDeviceToHost, s) != cudaSuccess ||
      cudaMemcpyAsync(b.h_bad, b.bad, 4, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
    set_err("plan readback failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  if (cudaStreamSynchronize(s) != cudaSuccess) {
    set_err("%s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  if (*b.h_bad) {
    set_err("request rejected: an id outside [-1, n_cold) for layer %d rows %d", layer, rows);
    return kRefuseCompute;
  }

  const at::Tensor x_h = dev_tensor(b.x_h, {rows, H}, at::kHalf, c.dev);
  const at::Tensor w_f = dev_tensor(b.w_f, {rows, T}, at::kFloat, c.dev);
  bool wrote = false;
  int total_max = 0;
  int total[QMC_MAX_GROUPS] = {0}, live[QMC_MAX_GROUPS] = {0};
  for (int g = 0; g < L.n_groups; g++) {
    live[g] = b.h_plan[L.n_groups + g];
    total[g] = b.h_plan[g] * bm[g];
    L.grp[g].last_live = live[g];
    if (total[g] > total_max) total_max = total[g];
  }
  int rc = 0;
  if (total_max <= 0) {
    /* No group has a live pick: the layer's answer for these rows is zero. */
    if (cudaMemsetAsync(b.acc, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
      set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseCompute;
    }
  } else {
    if (total_max > b.cap_sorted) {
      set_err("alignment needs %d rows, capacity %d", total_max, b.cap_sorted);
      return kRefuseMem;
    }
    /* The trio is sized for the largest group of this piece. I is the same for every group of a
     * layer (one trellis tile width), so taking it from group 0 is exact. */
    if (!grow_trio(c, total_max, L.grp[0].g.inter)) {
      set_err("workspace alloc failed: %s", cudaGetErrorString(cudaGetLastError()));
      return kRefuseMem;
    }
    for (int g = 0; g < L.n_groups && rc == 0; g++) {
      const qmc_group& q = L.grp[g].g;
      if (live[g] == 0) continue;
      int32_t* blk_off_g = b.blk_off + (size_t)g * QMC_MAX_EXPERTS;
      int32_t* eids_g = b.eids + (size_t)g * b.cap_sorted;
      if (cudaMemsetAsync(b.cursor + (size_t)g * QMC_MAX_EXPERTS, 0, (size_t)q.n_experts * 4, s) !=
              cudaSuccess ||
          cudaMemsetAsync(b.n_rows + g, 0, 4, s) != cudaSuccess) {
        set_err("alignment clear failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseCompute;
      }
      qmc_k_fill_i32(b.sorted, m_valid, total[g], s);
      qmc_k_scatter(b.ids, m_valid, q.n_cold, q.remap, b.cursor + (size_t)g * QMC_MAX_EXPERTS,
                    blk_off_g, bm[g], b.sorted, b.n_rows + g, s);
      if (c.dump) {
        L.dump_rows = rows;
        L.dump_bm[g] = bm[g];
        L.dump_nblk[g] = b.h_plan[g];
        L.dump_total[g] = total[g];
        if (cudaMemcpyAsync(b.h_sorted + (size_t)g * b.cap_sorted, b.sorted, (size_t)total[g] * 4,
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(b.h_eids + (size_t)g * b.cap_sorted, eids_g, (size_t)b.h_plan[g] * 4,
                            cudaMemcpyDeviceToHost, s) != cudaSuccess) {
          set_err("alignment dump failed: %s", cudaGetErrorString(cudaGetLastError()));
          return kRefuseCompute;
        }
      }

      const at::Tensor sorted_t = dev_tensor(b.sorted, {total[g]}, at::kInt, c.dev);
      const at::Tensor eids_t = dev_tensor(eids_g, {b.h_plan[g]}, at::kInt, c.dev);
      const at::Tensor nr_t = dev_tensor(b.n_rows + g, {1}, at::kInt, c.dev);
      const at::Tensor a13 = dev_tensor(b.a13, {2, total[g], H}, at::kHalf, c.dev);
      const at::Tensor inter =
          dev_tensor(b.inter, {total[g], 2 * (int64_t)q.inter}, at::kHalf, c.dev);
      const at::Tensor a2 = dev_tensor(b.a2, {1, total[g], (int64_t)q.inter}, at::kHalf, c.dev);
      const at::Tensor acc = dev_tensor(b.acc, {rows, H}, at::kFloat, c.dev);

      try {
        moe_had_in(x_h, a13, L.grp[g].suh13, sorted_t, eids_t, nr_t, bm[g], T, m_valid);
        moe_gemm(a13, L.grp[g].t13, L.grp[g].suh13, L.grp[g].svh13, eids_t, nr_t,
                 {(int64_t)q.inter, (int64_t)q.inter}, q.cb, bm[g], at::kHalf, sorted_t, true,
                 at::Tensor(), false, rows, T, inter, true, false);
        moe_glu(inter, a2, L.grp[g].suh2, eids_t, nr_t, bm[g]);
        moe_gemm(a2, L.grp[g].t2, L.grp[g].suh2, L.grp[g].svh2, eids_t, nr_t, {(int64_t)H}, q.cb,
                 bm[g], at::kFloat, sorted_t, true, w_f, true, rows, T, acc, true, wrote);
      } catch (const std::exception& e) {
        set_err("layer %d group %d raised: %s", layer, g, e.what());
        rc = kRefuseCompute;
        break;
      }
      wrote = true;
      if (c.trace)
        fprintf(stderr, "SEAM_MOE_CPP_PIECE layer=%d rows=%d g=%d E=%d bm=%d blk=%d pad=%d live=%d\n",
                layer, rows, g, q.n_experts, bm[g], b.h_plan[g], total[g], live[g]);
    }
    if (rc == 0 && !wrote) {
      if (cudaMemsetAsync(b.acc, 0, (size_t)rows * H * 4, s) != cudaSuccess) {
        set_err("zero fill failed: %s", cudaGetErrorString(cudaGetLastError()));
        return kRefuseCompute;
      }
    }
  }
  if (rc != 0) return rc;
  /* Answer -> pinned response slot. The Python path built this with .float().cpu().numpy(); it is
   * one D2H of the accumulator. With SEAM_MOE_CPP_OUT=bf16|fp16 the answer is rounded on device
   * first and only half the bytes cross the wire: the response is 5120 floats per row, two thirds
   * of a served request's payload, so this is the one place a wire format can be halved without
   * touching the kernels. The slot must then be half-sized - that is the seam's layout decision
   * (qes.py off_resp stride and the client's collect), not this file's. */
  const size_t obytes = (size_t)rows * H * (c.out_mode ? 2 : 4);
  const void* src = b.acc;
  if (c.out_mode) {
    qmc_k_f32_to_half(b.acc, b.out16, rows * H, c.out_mode, s);
    src = b.out16;
  }
  if (cudaMemcpyAsync(outp, src, obytes, cudaMemcpyDeviceToHost, s) != cudaSuccess) {
    set_err("D2H to the response slot failed: %s", cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  return 0;
}

}  // namespace

extern "C" int qmc_open(int device, int hidden, int topk, int ws_rows, void** out) {
  if (!out) return -1;
  *out = nullptr;
  auto* c = new Ctx();
  c->dev = device;
  c->hidden = hidden;
  c->topk = topk;
  c->ws_rows = ws_rows;
  c->ws_auto = env_int("SEAM_WS_ROWS_AUTO", 1) != 0;
  c->ws_guard = env_int("SEAM_WS_GUARD_MIB", 96);
  c->ws_hyst = env_int("SEAM_WS_HYST_MIB", 128);
  c->swap_warn = env_int("SEAM_SWAP_WARN_MIB", 512);
  c->fwd_floor = env_int("SEAM_FWD_FLOOR_MIB", 0);
  c->block_m = env_int("SEAM_MOE_CPP_BLOCK_M", 0);
  c->dump = env_int("SEAM_MOE_CPP_DUMP", 0) != 0;
  c->trace = env_int("SEAM_MOE_CPP_TRACE", 0) != 0;
  /* Response format. Default fp32: the shipped seam slot is fp32 and qes.py's off_resp stride is
   * rows*HIDDEN*4, so anything else must be a deliberate, matching change on the client side. */
  {
    const char* om = getenv("SEAM_MOE_CPP_OUT");
    if (om && (!strcmp(om, "bf16") || !strcmp(om, "bfloat16")))
      c->out_mode = 1;
    else if (om && (!strcmp(om, "fp16") || !strcmp(om, "f16") || !strcmp(om, "half")))
      c->out_mode = 2;
    else if (om && *om && strcmp(om, "fp32") && strcmp(om, "f32"))
      fprintf(stderr, "SEAM_MOE_CPP_WARN unknown SEAM_MOE_CPP_OUT=%s, keeping fp32\n", om);
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    set_err("cudaSetDevice(%d): %s", device, cudaGetErrorString(cudaGetLastError()));
    delete c;
    return -1;
  }
  if (cudaEventCreateWithFlags(&c->ev0, cudaEventDefault) != cudaSuccess ||
      cudaEventCreateWithFlags(&c->ev1, cudaEventDefault) != cudaSuccess) {
    set_err("event create failed: %s", cudaGetErrorString(cudaGetLastError()));
    delete c;
    return -1;
  }
  *out = c;
  return 0;
}

extern "C" int qmc_add_layer(void* ctx, int layer, int n_groups, const qmc_group* groups) {
  auto* c = (Ctx*)ctx;
  if (!c || layer < 0 || layer >= QMC_NL || n_groups <= 0 || n_groups > QMC_MAX_GROUPS || !groups) {
    set_err("qmc_add_layer: bad arguments (layer=%d n_groups=%d)", layer, n_groups);
    return -1;
  }
  if (n_groups > c->n_groups_max) c->n_groups_max = n_groups;
  Layer& L = c->layers[layer];
  const int H = c->hidden;
  L.n_groups = n_groups;
  for (int g = 0; g < n_groups; g++) {
    const qmc_group& q = groups[g];
    L.grp[g].g = q;
    if (q.n_experts <= 0 || q.n_experts > QMC_MAX_EXPERTS || q.inter <= 0 || (q.inter & 15) ||
        q.words13 <= 0 || q.words2 <= 0 || (H & 15) || q.n_cold <= 0) {
      set_err("qmc_add_layer: layer %d group %d geometry (E=%d I=%d w13=%d w2=%d H=%d n_cold=%d)",
              layer, g, q.n_experts, q.inter, q.words13, q.words2, H, q.n_cold);
      return -1;
    }
    const int64_t E = q.n_experts, h16 = H / 16, i16 = q.inter / 16;
    const at::TensorOptions opt[6] = {
        at::TensorOptions().dtype(at::kShort).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kShort).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev),
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, c->dev)};
    L.grp[g].t13 = at::from_blob((void*)q.trellis13, {E, h16, 2 * i16, q.words13}, opt[0]);
    L.grp[g].suh13 = at::from_blob((void*)q.suh13, {E, 2, H}, opt[1]);
    L.grp[g].svh13 = at::from_blob((void*)q.svh13, {E, 32 * i16}, opt[2]);
    L.grp[g].t2 = at::from_blob((void*)q.trellis2, {E, i16, h16, q.words2}, opt[3]);
    L.grp[g].suh2 = at::from_blob((void*)q.suh2, {E, 1, q.inter}, opt[4]);
    L.grp[g].svh2 = at::from_blob((void*)q.svh2, {E, H}, opt[5]);
  }
  L.have = true;
  return 0;
}

extern "C" int qmc_compute(void* ctx, int layer, int rows, const void* x_bf16, const int32_t* ids,
                           const void* w_bf16, float* out_f32) {
  auto* c = (Ctx*)ctx;
  if (!c || !x_bf16 || !ids || !w_bf16 || !out_f32) return kRefuseCompute;
  if (cudaSetDevice(c->dev) != cudaSuccess) {
    set_err("cudaSetDevice(%d): %s", c->dev, cudaGetErrorString(cudaGetLastError()));
    return kRefuseCompute;
  }
  if (layer < 0 || layer >= QMC_NL || !c->layers[layer].have) {
    set_err("layer %d has no registered cuda-exl3 groups", layer);
    return kRefuseCompute;
  }
  if (rows <= 0 || rows > QMC_MAX_ROWS) {
    set_err("rows=%d out of range", rows);
    return kRefuseCompute;
  }
  std::lock_guard<std::mutex> lk(c->mu);
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  c->calls++;

  int avail = avail_mib();
  if (c->fwd_floor > 0 && avail >= 0 && avail < c->fwd_floor) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    avail = avail_mib();
  }
  if (c->fwd_floor > 0 && avail >= 0 && avail < c->fwd_floor) {
    c->refused++;
    fprintf(stderr,
            "SEAM_COMPUTE_REFUSED layer=%d rows=%d why=memory need_mib=%d avail_mib=%d "
            "swapped_mib=%d calls=%lld refusals=%lld - answering nothing, the session ends, the "
            "server keeps listening\n",
            layer, rows, c->fwd_floor, avail, swap_used_mib(), (long long)c->calls,
            (long long)c->refused);
    fflush(stderr);
    return kRefuseMem;
  }

  cudaStream_t s = at::cuda::getCurrentCUDAStream(c->dev).stream();
  cudaEventRecord(c->ev0, s);
  int rc = 0;
  const int step = pick_step(*c, rows);
  for (int r0 = 0; r0 < rows && rc == 0; r0 += step) {
    const int n = std::min(step, rows - r0);
    rc = run_piece(*c, layer, n, (const uint8_t*)x_bf16 + (size_t)r0 * c->hidden * 2,
                   ids + (size_t)r0 * c->topk,
                   (const uint8_t*)w_bf16 + (size_t)r0 * c->topk * 2,
                   out_f32 + (size_t)r0 * c->hidden);
  }
  cudaEventRecord(c->ev1, s);
  const cudaError_t se = cudaStreamSynchronize(s);
  if (rc == 0 && se != cudaSuccess) {
    set_err("stream sync: %s", cudaGetErrorString(se));
    rc = kRefuseCompute;
  }
  float dev_ms = 0.f;
  if (cudaEventElapsedTime(&dev_ms, c->ev0, c->ev1) != cudaSuccess) dev_ms = -1.f;
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  c->last_wall = (int64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000LL + (ts1.tv_nsec - ts0.tv_nsec);
  c->last_dev = (int64_t)(dev_ms * 1e6);
  if (rc != 0) {
    c->refused++;
    const bool usable = cuda_alive();
    fprintf(stderr,
            "SEAM_COMPUTE_%s layer=%d rows=%d err=%s avail_mib=%d swapped_mib=%d calls=%lld "
            "refusals=%lld cuda_ok=%d\n",
            usable ? "REFUSED" : "FAIL", layer, rows, g_err.c_str(), avail_mib(), swap_used_mib(),
            (long long)c->calls, (long long)c->refused, (int)usable);
    fflush(stderr);
    /* worker_seam_server's fail-stop exits 70 on the same condition; here the caller sees the
     * refusal code and its own health probe decides. */
  }
  if (!c->reported) {
    c->reported = true;
    int nlayers = 0;
    for (auto& l : c->layers) nlayers += (int)l.have;
    fprintf(stderr,
            "SEAM_MOE_CPP ready=1 device=%d hidden=%d topk=%d ws_rows=%d ws_auto=%d layers=%d "
            "groups=%d out=%s\n",
            c->dev, c->hidden, c->topk, c->ws_rows, (int)c->ws_auto, nlayers,
            c->n_groups_max, c->out_mode == 1 ? "bf16" : (c->out_mode == 2 ? "fp16" : "fp32"));
    fflush(stderr);
  }
  return rc;
}

extern "C" int qmc_last_call_ns(void* ctx, int64_t* wall_ns, int64_t* dev_ns) {
  auto* c = (Ctx*)ctx;
  if (!c) return -1;
  if (wall_ns) *wall_ns = c->last_wall;
  if (dev_ns) *dev_ns = c->last_dev;
  return 0;
}

extern "C" int qmc_stats(void* ctx, int64_t* calls, int64_t* refused) {
  auto* c = (Ctx*)ctx;
  if (!c) return -1;
  if (calls) *calls = c->calls;
  if (refused) *refused = c->refused;
  return 0;
}

extern "C" int qmc_out_mode(void* ctx) {
  auto* c = (Ctx*)ctx;
  return c ? c->out_mode : -1;
}

extern "C" int qmc_set_out_mode(void* ctx, int mode) {
  auto* c = (Ctx*)ctx;
  if (!c || mode < 0 || mode > 2) return -1;
  std::lock_guard<std::mutex> lk(c->mu);
  const int old = c->out_mode;
  c->out_mode = mode;
  return old;
}

/* Alignment knobs. block_m = 0 keeps the automatic tier (previous call's live count); a positive
 * value freezes it, which is what a captured graph needs - the tier may not change shape inside a
 * capture. */
extern "C" int qmc_set_block_m(void* ctx, int block_m) {
  auto* c = (Ctx*)ctx;
  if (!c || block_m < 0) return -1;
  std::lock_guard<std::mutex> lk(c->mu);
  const int old = c->block_m;
  c->block_m = block_m;
  return old;
}

/* Debug: copy the alignment of the last piece served for `layer` back to the caller, so the bench
 * can compare it with align_block() for the same ids. Returns 0, or -1 with qmc_last_error() set;
 * *nblk/*total are 0 when the layer has no group `group` or nothing was served. */
extern "C" int qmc_dump_align(void* ctx, int layer, int group, int32_t* sorted, int32_t* eids,
                              int* nblk, int* total, int* block_m) {
  auto* c = (Ctx*)ctx;
  if (!c || !nblk || !total || !block_m) return -1;
  *nblk = *total = *block_m = 0;
  if (!c->dump) {
    set_err("qmc_dump_align: SEAM_MOE_CPP_DUMP is off");
    return -1;
  }
  if (layer < 0 || layer >= QMC_NL || group < 0 || group >= QMC_MAX_GROUPS) {
    set_err("qmc_dump_align: bad layer %d group %d", layer, group);
    return -1;
  }
  Layer& L = c->layers[layer];
  Buffers& b = c->b;
  if (group >= L.n_groups || L.dump_rows == 0) return 0;
  *nblk = L.dump_nblk[group];
  *total = L.dump_total[group];
  *block_m = L.dump_bm[group];
  if (sorted && *total > 0)
    memcpy(sorted, b.h_sorted + (size_t)group * b.cap_sorted, (size_t)*total * 4);
  if (eids && *nblk > 0)
    memcpy(eids, b.h_eids + (size_t)group * b.cap_sorted, (size_t)*nblk * 4);
  return 0;
}

extern "C" const char* qmc_last_error(void) { return g_err.c_str(); }

extern "C" void qmc_close(void* ctx) {
  auto* c = (Ctx*)ctx;
  if (!c) return;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    for (auto& l : c->layers) {
      for (auto& g : l.grp) {
        g.t13 = at::Tensor();
        g.suh13 = at::Tensor();
        g.svh13 = at::Tensor();
        g.t2 = at::Tensor();
        g.suh2 = at::Tensor();
        g.svh2 = at::Tensor();
      }
    }
    if (c->ev0) cudaEventDestroy(c->ev0);
    if (c->ev1) cudaEventDestroy(c->ev1);
    c->ev0 = c->ev1 = nullptr;
  }
  delete c; /* Buffers' destructor frees the rest */
}
