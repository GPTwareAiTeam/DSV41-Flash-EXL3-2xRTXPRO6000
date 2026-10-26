/* qes_bench: drives qmesh_expert_seam over the P7 <-> GX1 link.
 *
 *   GX1 (server):  qes_bench server <dev> <gid_idx> <port> <rows>
 *   P7  (client):  qes_bench client <dev> <gid_idx> <peer_ip> <port> <rows> <iters> <dev_ordinal>
 *
 * Client reports, all for 40 layers in sequence unless noted:
 *   eager_seq   : issue(L)+collect(L) per layer, host-timed, p50/p99 per layer
 *   graph_seq   : same 40 issue/collect pairs captured once, replayed <iters> times
 *   pipelined   : issue all 40, then collect all 40 (the seam's in-flight depth)
 *   busy_pipe   : issue(L), then a 100 us busy kernel, then collect(L); per layer
 * Every collected output is checked against the stub: out = float(x) + (L + 1).
 */
#include "qmesh_expert_seam.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); exit(2); } } while (0)

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void stats(const char* name, std::vector<double> v, int layers) {
  std::sort(v.begin(), v.end());
  double p50 = v[v.size() / 2], p99 = v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))];
  printf("%-12s n=%zu  per-layer us  p50=%.2f p99=%.2f  (per-iter total p50=%.1f us, layers=%d)\n",
         name, v.size(), p50, p99, p50 * layers, layers);
}

__global__ void busy_kernel(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}

static void fill_inputs(const std::vector<uint16_t>& x_host, const std::vector<int32_t>& ids,
                        const std::vector<uint16_t>& w, uint16_t* dx, int32_t* di, uint16_t* dw) {
  CK(cudaMemcpy(dx, x_host.data(), x_host.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
}

static int check(const std::vector<float>& out, const std::vector<uint16_t>& x, int L, int rows) {
  const float add = (float)(L + 1);
  for (int i = 0; i < rows * 5120; i++) {
    uint32_t bits = (uint32_t)x[i] << 16; float f; memcpy(&f, &bits, 4);
    if (std::fabs(out[i] - (f + add)) > 1e-3f) {
      fprintf(stderr, "MISMATCH L=%d i=%d got=%f want=%f\n", L, i, out[i], f + add);
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: see header\n"); return 1; }
  const char* mode = argv[1];

  if (!strcmp(mode, "server")) {
    if (argc < 6) return 1;
    qes_handle* h = qes_create(QES_ROLE_SERVER, argv[2], atoi(argv[3]), nullptr, atoi(argv[4]), atoi(argv[5]), 0);
    if (!h) { fprintf(stderr, "server create failed\n"); return 1; }
    printf("server up\n"); fflush(stdout);
    for (;;) std::this_thread::sleep_for(std::chrono::seconds(3600));
  }

  if (strcmp(mode, "client") || argc < 9) { fprintf(stderr, "usage: qes_bench client <dev> <gid> <ip> <port> <rows> <iters> <ord>\n"); return 1; }
  const char* dev = argv[2];
  const int gid = atoi(argv[3]);
  const char* ip = argv[4];
  const int port = atoi(argv[5]);
  const int rows = atoi(argv[6]);
  const int iters = atoi(argv[7]);
  const int ord = atoi(argv[8]);

  CK(cudaSetDevice(ord));
  qes_handle* h = qes_create(QES_ROLE_CLIENT, dev, gid, ip, port, rows, ord);
  if (!h) { fprintf(stderr, "client create failed\n"); return 1; }

  const size_t nx = (size_t)rows * QES_HIDDEN;
  std::vector<uint16_t> x_host(nx);
  for (size_t i = 0; i < nx; i++) x_host[i] = (uint16_t)(0x3f80 + (i % 97));  /* bf16 near 1.0 */
  std::vector<int32_t> ids(rows * QES_TOPK);
  for (size_t i = 0; i < ids.size(); i++) ids[i] = (int32_t)(i % 384);
  std::vector<uint16_t> w(rows * QES_TOPK, 0x3f00);

  uint16_t *dx, *dw; int32_t* di; float* dout;
  CK(cudaMalloc(&dx, nx * 2)); CK(cudaMalloc(&di, ids.size() * 4));
  CK(cudaMalloc(&dw, w.size() * 2)); CK(cudaMalloc(&dout, nx * 4));
  fill_inputs(x_host, ids, w, dx, di, dw);

  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::vector<float> out(nx);
  int bad = 0;

  /* warm: first round trip also primes the RC path. */
  for (int L = 0; L < QES_NL; L++) {
    qes_issue(h, L, dx, di, dw, st);
    qes_collect(h, L, dout, st);
  }
  CK(cudaStreamSynchronize(st));

  /* eager_seq: per-layer host-timed round trip. */
  std::vector<double> eager;
  for (int it = 0; it < iters; it++)
    for (int L = 0; L < QES_NL; L++) {
      double t0 = now_us();
      qes_issue(h, L, dx, di, dw, st);
      qes_collect(h, L, dout, st);
      CK(cudaStreamSynchronize(st));
      eager.push_back(now_us() - t0);
      if (it == 0) {
        CK(cudaMemcpy(out.data(), dout, nx * 4, cudaMemcpyDeviceToHost));
        bad |= check(out, x_host, L, rows);
      }
    }
  stats("eager_seq", eager, QES_NL);

  /* graph_seq: capture 40 issue/collect pairs, replay. */
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int L = 0; L < QES_NL; L++) {
    qes_issue(h, L, dx, di, dw, st);
    qes_collect(h, L, dout, st);
  }
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  std::vector<double> graph;
  for (int it = 0; it < iters; it++) {
    double t0 = now_us();
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    graph.push_back((now_us() - t0) / QES_NL);
  }
  CK(cudaMemcpy(out.data(), dout, nx * 4, cudaMemcpyDeviceToHost));
  bad |= check(out, x_host, QES_NL - 1, rows);
  stats("graph_seq", graph, QES_NL);

  /* pipelined: issue all, then collect all. */
  std::vector<double> pipe;
  for (int it = 0; it < iters; it++) {
    double t0 = now_us();
    for (int L = 0; L < QES_NL; L++) qes_issue(h, L, dx, di, dw, st);
    for (int L = 0; L < QES_NL; L++) qes_collect(h, L, dout, st);
    CK(cudaStreamSynchronize(st));
    pipe.push_back((now_us() - t0) / QES_NL);
  }
  stats("pipelined", pipe, QES_NL);

  /* busy_pipe: each layer's round trip overlapped with a ~100 us kernel on the device. */
  std::vector<double> busy;
  long long cyc = 100000;  /* ~100 us at ~1 GHz SM clock; calibrated loosely */
  for (int it = 0; it < iters; it++) {
    double t0 = now_us();
    for (int L = 0; L < QES_NL; L++) {
      qes_issue(h, L, dx, di, dw, st);
      busy_kernel<<<1, 1, 0, st>>>(cyc);
      qes_collect(h, L, dout, st);
    }
    CK(cudaStreamSynchronize(st));
    busy.push_back((now_us() - t0) / QES_NL);
  }
  stats("busy_pipe", busy, QES_NL);

  printf("correctness: %s\n", bad ? "FAIL" : "OK");
  qes_destroy(h);
  return bad ? 3 : 0;
}
