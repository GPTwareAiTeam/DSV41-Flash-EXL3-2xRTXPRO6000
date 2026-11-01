/* qmesh_expert_seam.cu: host-staged RC RDMA expert seam, see qmesh_expert_seam.h.
 *
 * Slab layout, identical on both peers, per layer L (stride bytes, 64B aligned):
 *   [req_off, +req_len)          request : x bf16[rows,5120] | ids i32[rows,6] | w bf16[rows,6]
 *   [resp_off, +resp_len)        response: out f32[rows,5120]
 *   F_REQ  (u64, own 64B line)   request seq. Client: GPU publish kernel bumps it.
 *                                Server: remote RDMA flag WRITE lands here.
 *   F_RESP (u64, own 64B line)   response seq. Server: RDMA flag WRITE from its proxy.
 *                                Client: remote RDMA flag WRITE lands here.
 *
 * Protocol per call (seq = number of issues on that layer, starting at 1):
 *   issue  : GPU D2H copies the request into the slab; publish kernel then bumps
 *            F_REQ with st.release.sys (stream order guarantees the copies landed).
 *            Client proxy sees F_REQ advance, posts RDMA WRITE(payload) unsignaled,
 *            then RDMA WRITE(inline seq -> server F_REQ) signaled.
 *   server : proxy sees its F_REQ advance, computes the response into its slab,
 *            posts RDMA WRITE(payload), then RDMA WRITE(inline seq -> client F_RESP).
 *   collect: wait kernel spins (ld.acquire.sys) on F_RESP >= F_REQ, then the
 *            response is H2D copied to the caller's device buffer.
 *
 * Ordering: RC delivers in order on one QP, so a flag observed by the peer implies
 * its preceding payload is placed. Inline flag WRs copy the seq at post time, so no
 * source word is reused while in flight. A layer's buffers are reused only after the
 * caller collected, which the decode loop guarantees.
 *
 * Compute stub on the server: out[r][j] = float(x[r][j]) + (L + 1). This verifies
 * the layer index and payload. Real expert compute replaces server_compute().
 */
#include "qmesh_expert_seam.h"

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace {

constexpr int ROWS_MAX = 64;
constexpr size_t LINE = 64;
constexpr uint64_t WAIT_TIMEOUT_NS = 30ull * 1000 * 1000 * 1000;

[[noreturn]] void fail(const char* what) {
  fprintf(stderr, "qes: %s (errno=%d %s)\n", what, errno, strerror(errno));
  abort();
}

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

/* Hello exchanged over the management TCP socket. */
struct Hello {
  uint32_t qpn, psn, lid, mtu;
  uint8_t gid[16];
  uint64_t base;
  uint32_t rkey;
  uint32_t layers, rows;
};

void xfer(int fd, void* buf, size_t n, bool send) {
  char* p = (char*)buf;
  while (n) {
    ssize_t k = send ? ::send(fd, p, n, 0) : ::recv(fd, p, n, 0);
    if (k <= 0) { if (k < 0 && errno == EINTR) continue; fail(send ? "send" : "recv"); }
    p += k; n -= (size_t)k;
  }
}

int tcp_listen_accept(int port) {
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  if (ls < 0) fail("socket");
  int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
  if (bind(ls, (sockaddr*)&a, sizeof(a)) || listen(ls, 1)) fail("bind/listen");
  int c = accept(ls, nullptr, nullptr);
  if (c < 0) fail("accept");
  close(ls);
  int one2 = 1; setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one2, sizeof(one2));
  return c;
}

int tcp_connect(const char* ip, int port) {
  for (int tries = 0; tries < 600; tries++) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, ip, &a.sin_addr);
    if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) {
      int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      return s;
    }
    close(s);
    usleep(100000);
  }
  fail("connect timeout");
}

/* ---- device kernels ---- */

/* Bump the request seq. Single writer, GPU side. release.sys orders it after the
 * request copies for the CPU proxy. */
__global__ void publish_kernel(uint64_t* seq) {
  uint64_t v = *seq + 1;
  asm volatile("st.release.sys.u64 [%0], %1;" :: "l"(seq), "l"(v) : "memory");
}

/* Spin until resp_seq >= req_seq. Traps after WAIT_TIMEOUT_NS instead of hanging. */
__global__ void wait_kernel(const uint64_t* req_seq, const uint64_t* resp_seq) {
  uint64_t want = *req_seq;
  uint64_t t0, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (true) {
    uint64_t got;
    asm volatile("ld.acquire.sys.u64 %0, [%1];" : "=l"(got) : "l"(resp_seq) : "memory");
    if (got >= want) break;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (now - t0 > WAIT_TIMEOUT_NS) __trap();
  }
}

}  // namespace

struct qes_handle {
  int role = 0, rows = 0, dev = 0, port = 1, gid_idx = 0;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  ibv_mr* mr = nullptr;
  ibv_port_attr port_attr{};
  uint8_t* slab = nullptr;  /* pinned, mapped host */
  size_t slab_bytes = 0, stride = 0, req_len = 0, resp_len = 0;
  size_t ids_off = 0, w_off = 0;
  uint64_t remote_base = 0;
  uint32_t remote_rkey = 0;
  std::atomic<bool> running{false};
  std::thread proxy;
  uint64_t handled[QES_NL] = {};  /* proxy-owned: last seq answered per layer */
  qes_server_fn server_fn = nullptr;  /* optional compute hook (server role) */
  void* server_ctx = nullptr;
};

namespace {

size_t off_req(const qes_handle* h, int L) { return (size_t)L * h->stride; }
size_t off_resp(const qes_handle* h, int L) { return off_req(h, L) + align_up(h->req_len, LINE); }
size_t off_freq(const qes_handle* h, int L) { return off_resp(h, L) + align_up(h->resp_len, LINE); }
size_t off_fresp(const qes_handle* h, int L) { return off_freq(h, L) + LINE; }

uint64_t load_acq(const uint8_t* base, size_t off) {
  return __atomic_load_n((const uint64_t*)(base + off), __ATOMIC_ACQUIRE);
}

void drain_cq(qes_handle* h) {
  ibv_wc wc[16];
  int n;
  while ((n = ibv_poll_cq(h->cq, 16, wc)) > 0) {
    for (int i = 0; i < n; i++)
      if (wc[i].status != IBV_WC_SUCCESS) {
        fprintf(stderr, "qes: wc error %s opcode=%d\n", ibv_wc_status_str(wc[i].status), (int)wc[i].opcode);
        abort();
      }
  }
  if (n < 0) fail("ibv_poll_cq");
}

/* RDMA WRITE of `len` bytes from local slab offset to the same offset remotely. */
void post_payload(qes_handle* h, size_t off, uint32_t len) {
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)(h->slab + off);
  sge.length = len;
  sge.lkey = h->mr->lkey;
  ibv_send_wr wr{};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr.rdma.remote_addr = h->remote_base + off;
  wr.wr.rdma.rkey = h->remote_rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(h->qp, &wr, &bad)) fail("ibv_post_send payload");
}

/* Inline (copied at post time) RDMA WRITE of an 8-byte seq to a remote flag word. */
void post_flag(qes_handle* h, size_t remote_off, uint64_t seq) {
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)&seq;
  sge.length = 8;
  ibv_send_wr wr{};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
  wr.wr.rdma.remote_addr = h->remote_base + remote_off;
  wr.wr.rdma.rkey = h->remote_rkey;
  ibv_send_wr* bad = nullptr;
  if (ibv_post_send(h->qp, &wr, &bad)) fail("ibv_post_send flag");
}

void server_compute(qes_handle* h, int L) {
  const float add = (float)(L + 1);
  const uint16_t* x = (const uint16_t*)(h->slab + off_req(h, L));
  float* out = (float*)(h->slab + off_resp(h, L));
  const int n = h->rows * QES_HIDDEN;
  for (int i = 0; i < n; i++) {
    uint32_t bits = (uint32_t)x[i] << 16;
    float f;
    memcpy(&f, &bits, 4);
    out[i] = f + add;
  }
}

/* Client proxy: forwards every new request seq on any layer. */
void client_proxy(qes_handle* h) {
  while (h->running.load(std::memory_order_relaxed)) {
    drain_cq(h);
    for (int L = 0; L < QES_NL; L++) {
      uint64_t r = load_acq(h->slab, off_freq(h, L));
      if (r > h->handled[L]) {
        post_payload(h, off_req(h, L), (uint32_t)h->req_len);
        post_flag(h, off_freq(h, L), r);
        h->handled[L] = r;
      }
    }
  }
}

/* Server proxy: computes and returns every new request seq on any layer. */
void server_proxy(qes_handle* h) {
  while (h->running.load(std::memory_order_relaxed)) {
    drain_cq(h);
    for (int L = 0; L < QES_NL; L++) {
      uint64_t r = load_acq(h->slab, off_freq(h, L));
      if (r > h->handled[L]) {
        if (h->server_fn) {
          h->server_fn(h->server_ctx, L, h->rows,
                       h->slab + off_req(h, L), (const int32_t*)(h->slab + off_req(h, L) + h->ids_off),
                       h->slab + off_req(h, L) + h->w_off, (float*)(h->slab + off_resp(h, L)));
        } else {
          server_compute(h, L);
        }
        post_payload(h, off_resp(h, L), (uint32_t)h->resp_len);
        post_flag(h, off_fresp(h, L), r);
        h->handled[L] = r;
      }
    }
  }
}

}  // namespace

extern "C" qes_handle* qes_create(int role, const char* dev_name, int gid_idx, const char* peer_ip,
                                  int port, int rows, int device_ordinal) {
  if (rows < 1 || rows > ROWS_MAX) return nullptr;
  qes_handle* h = new qes_handle();
  h->role = role; h->rows = rows; h->dev = device_ordinal; h->gid_idx = gid_idx;
  h->port = 1;
  h->req_len = (size_t)rows * (QES_HIDDEN * 2 + QES_TOPK * 4 + QES_TOPK * 2);
  h->ids_off = (size_t)rows * QES_HIDDEN * 2;
  h->w_off = h->ids_off + (size_t)rows * QES_TOPK * 4;
  h->resp_len = (size_t)rows * QES_HIDDEN * 4;
  h->stride = align_up(h->req_len, LINE) + align_up(h->resp_len, LINE) + 2 * LINE;
  h->slab_bytes = h->stride * QES_NL;

  if (cudaSetDevice(h->dev) != cudaSuccess) fail("cudaSetDevice");
  if (cudaHostAlloc(&h->slab, h->slab_bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
    fail("cudaHostAlloc");
  memset(h->slab, 0, h->slab_bytes);

  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n <= 0) fail("ibv_get_device_list");
  ibv_device* dev = nullptr;
  for (int i = 0; i < n; i++)
    if (strcmp(ibv_get_device_name(list[i]), dev_name) == 0) dev = list[i];
  if (!dev) fail("IB device not found");
  h->ctx = ibv_open_device(dev);
  ibv_free_device_list(list);
  if (!h->ctx) fail("ibv_open_device");
  if (ibv_query_port(h->ctx, h->port, &h->port_attr)) fail("ibv_query_port");
  h->pd = ibv_alloc_pd(h->ctx); if (!h->pd) fail("ibv_alloc_pd");
  h->cq = ibv_create_cq(h->ctx, 1024, nullptr, nullptr, 0); if (!h->cq) fail("ibv_create_cq");

  ibv_qp_init_attr qi{};
  qi.send_cq = h->cq; qi.recv_cq = h->cq; qi.qp_type = IBV_QPT_RC;
  qi.cap.max_send_wr = 512; qi.cap.max_recv_wr = 8;
  qi.cap.max_send_sge = 1; qi.cap.max_recv_sge = 1;
  qi.cap.max_inline_data = 64;
  h->qp = ibv_create_qp(h->pd, &qi); if (!h->qp) fail("ibv_create_qp");

  const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  h->mr = ibv_reg_mr(h->pd, h->slab, h->slab_bytes, access);
  if (!h->mr) fail("ibv_reg_mr(slab)");

  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = h->port; a.qp_access_flags = access;
  if (ibv_modify_qp(h->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
    fail("qp init");

  union ibv_gid lg{};
  if (ibv_query_gid(h->ctx, h->port, gid_idx, &lg)) fail("ibv_query_gid");
  const uint32_t my_psn = (uint32_t)(lrand48() & 0xffffff);

  Hello mine{};
  mine.qpn = h->qp->qp_num; mine.psn = my_psn; mine.lid = h->port_attr.lid;
  mine.mtu = h->port_attr.active_mtu;
  memcpy(mine.gid, lg.raw, 16);
  mine.base = (uint64_t)(uintptr_t)h->slab;
  mine.rkey = h->mr->rkey;
  mine.layers = QES_NL; mine.rows = (uint32_t)rows;

  int sock = role == QES_ROLE_SERVER ? tcp_listen_accept(port) : tcp_connect(peer_ip, port);
  Hello peer{};
  /* The server answers only after its QP is in RTS (below): the client starts writing as soon as it has
   * our hello, and an RC write to a responder that is still in INIT/RTR is NAKed as "remote invalid
   * request" and pushes our QP to ERROR (seen as qes: wc error ... opcode=1 on the client). */
  if (role == QES_ROLE_SERVER) { xfer(sock, &peer, sizeof(peer), false); }
  else { xfer(sock, &mine, sizeof(mine), true); xfer(sock, &peer, sizeof(peer), false); close(sock); }
  if (peer.layers != QES_NL || peer.rows != (uint32_t)rows) fail("peer layout mismatch");

  /* RTR: destination GID is the peer's RoCEv2 GID; sgid_index is our local index. */
  ibv_qp_attr r{};
  r.qp_state = IBV_QPS_RTR;
  r.path_mtu = h->port_attr.active_mtu;
  r.dest_qp_num = peer.qpn; r.rq_psn = peer.psn;
  r.max_dest_rd_atomic = 4; r.min_rnr_timer = 12;
  r.ah_attr.is_global = 1; r.ah_attr.port_num = (uint8_t)h->port;
  r.ah_attr.dlid = (uint16_t)peer.lid; r.ah_attr.sl = 0; r.ah_attr.src_path_bits = 0;
  memcpy(r.ah_attr.grh.dgid.raw, peer.gid, 16);
  r.ah_attr.grh.sgid_index = (uint8_t)gid_idx;
  r.ah_attr.grh.hop_limit = 1;
  if (ibv_modify_qp(h->qp, &r, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
    fail("qp rtr");
  ibv_qp_attr s{};
  s.qp_state = IBV_QPS_RTS; s.sq_psn = my_psn; s.timeout = 14; s.retry_cnt = 7; s.rnr_retry = 7;
  s.max_rd_atomic = 4;
  if (ibv_modify_qp(h->qp, &s, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC))
    fail("qp rts");
  if (role == QES_ROLE_SERVER) { xfer(sock, &mine, sizeof(mine), true); close(sock); }

  h->remote_base = peer.base;
  h->remote_rkey = peer.rkey;

  h->running = true;
  h->proxy = std::thread(role == QES_ROLE_SERVER ? server_proxy : client_proxy, h);
  return h;
}

extern "C" int qes_issue(qes_handle* h, int layer, const void* x_bf16, const int32_t* ids,
                         const void* w_bf16, cudaStream_t stream) {
  if (!h || h->role != QES_ROLE_CLIENT || layer < 0 || layer >= QES_NL) return -1;
  uint8_t* req = h->slab + off_req(h, layer);
  if (cudaMemcpyAsync(req, x_bf16, (size_t)h->rows * QES_HIDDEN * 2, cudaMemcpyDeviceToHost, stream)) return -2;
  if (cudaMemcpyAsync(req + h->ids_off, ids, (size_t)h->rows * QES_TOPK * 4, cudaMemcpyDeviceToHost, stream)) return -2;
  if (cudaMemcpyAsync(req + h->w_off, w_bf16, (size_t)h->rows * QES_TOPK * 2, cudaMemcpyDeviceToHost, stream)) return -2;
  publish_kernel<<<1, 1, 0, stream>>>((uint64_t*)(h->slab + off_freq(h, layer)));
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" int qes_collect(qes_handle* h, int layer, float* out_f32, cudaStream_t stream) {
  if (!h || h->role != QES_ROLE_CLIENT || layer < 0 || layer >= QES_NL) return -1;
  wait_kernel<<<1, 1, 0, stream>>>((const uint64_t*)(h->slab + off_freq(h, layer)),
                                   (const uint64_t*)(h->slab + off_fresp(h, layer)));
  if (cudaMemcpyAsync(out_f32, h->slab + off_resp(h, layer), h->resp_len, cudaMemcpyHostToDevice, stream)) return -2;
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" void qes_set_server_fn(qes_handle* h, qes_server_fn fn, void* ctx) {
  if (!h || h->role != QES_ROLE_SERVER) return;
  h->server_ctx = ctx;
  h->server_fn = fn;  /* set before the first client request; the proxy reads it per request */
}

extern "C" void qes_destroy(qes_handle* h) {
  if (!h) return;
  h->running = false;
  if (h->proxy.joinable()) h->proxy.join();
  if (h->qp) ibv_destroy_qp(h->qp);
  if (h->mr) ibv_dereg_mr(h->mr);
  if (h->cq) ibv_destroy_cq(h->cq);
  if (h->pd) ibv_dealloc_pd(h->pd);
  if (h->ctx) ibv_close_device(h->ctx);
  if (h->slab) cudaFreeHost(h->slab);
  delete h;
}
