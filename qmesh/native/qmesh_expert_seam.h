/* qmesh_expert_seam: P7 <-> GX1 expert-offload seam over host-staged RC RDMA.
 *
 * Client (role 0, P7) issues per-layer requests from CUDA streams/graphs:
 *   qes_issue(h, L, x, ids, w, stream)
 *       x:   bf16 [rows,5120] device ptr
 *       ids: int32 [rows,6]  device ptr (cold-local expert ids)
 *       w:   bf16 [rows,6]   device ptr (routing weights)
 *   qes_collect(h, L, out, stream)
 *       out: fp32 [rows,5120] device ptr (partial result from the server)
 * Both are kernel launches; sequence counters live in mapped host memory, so
 * the calls are capturable in CUDA graphs. Each layer L owns one slot.
 *
 * Server (role 1, GX1) runs a proxy thread that answers requests. The compute
 * stub is out = float(x) + 1 (verifiable). Real expert compute replaces it.
 *
 * Wire, per layer and direction: RC RDMA WRITE of the payload, then an 8-byte
 * RDMA WRITE of the sequence flag on the same QP. RC delivers in order, so a
 * flag observed by the peer implies its payload is visible.
 * Memory: pinned host memory (cudaHostAllocMapped). No GPUDirect RDMA: it is
 * blocked on both hosts (nvidia-peermem absent on P7, ibv_reg_mr EFAULT on GX1).
 */
#ifndef QMESH_EXPERT_SEAM_H
#define QMESH_EXPERT_SEAM_H

#include <cuda_runtime.h>
#include <stdint.h>

#define QES_NL 40
#define QES_HIDDEN 5120
#define QES_TOPK 6

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qes_handle qes_handle;

#define QES_ROLE_CLIENT 0
#define QES_ROLE_SERVER 1

/* Client connects to peer_ip:port. Server listens on port (peer_ip ignored).
 * rows is fixed per handle and sizes the wire payload. */
qes_handle* qes_create(int role, const char* dev_name, int gid_idx,
                       const char* peer_ip, int port, int rows, int device_ordinal);

/* Optional server-side compute hook. When set on a server handle, the proxy thread calls it
 * instead of the built-in stub. Pointers are into the pinned slab for layer L: x bf16
 * [rows,QES_HIDDEN], ids int32 [rows,QES_TOPK] (cold-local, -1 = not on this host),
 * w bf16 [rows,QES_TOPK], out f32 [rows,QES_HIDDEN] (must be fully written on return). */
typedef void (*qes_server_fn)(void* ctx, int layer, int rows, const void* x_bf16,
                              const int32_t* ids, const void* w_bf16, float* out_f32);
void qes_set_server_fn(qes_handle* h, qes_server_fn fn, void* ctx);

int qes_issue(qes_handle* h, int layer, const void* x_bf16, const int32_t* ids,
              const void* w_bf16, cudaStream_t stream);
int qes_collect(qes_handle* h, int layer, float* out_f32, cudaStream_t stream);

void qes_destroy(qes_handle* h);

#ifdef __cplusplus
}
#endif

#endif
