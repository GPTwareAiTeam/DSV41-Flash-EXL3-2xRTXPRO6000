"""Null remote-expert host for P7-only timing. Same surface as qmesh.remote_host.RemoteExpertHost,
but the 'cold' partial is a zero tensor on the GPU: no network, no sync, no GX1 compute.
Install with install() BEFORE model load (block_sparse_mlp_cpu imports the class at load time)."""
import torch


class NullRemoteExpertHost:
    @classmethod
    def for_config(cls, infer_params):
        hosts = getattr(infer_params, "remote_expert_hosts", None)
        if hosts is None:
            hosts = {}
            infer_params.remote_expert_hosts = hosts
        if "null" not in hosts:
            hosts["null"] = cls()
        return hosts["null"]

    def __init__(self):
        self.stream_min_rows = 10 ** 9
        self.layer_index = {}

    def register_layer(self, key, *args, **kwargs):
        idx = int(key.split(".")[1])
        self.layer_index[key] = idx
        return idx

    def submit_issue(self, layer_idx, y, selected, weights):
        return torch.zeros(y.shape[:-1] + (y.shape[-1],), dtype = torch.float32, device = y.device)

    def submit_collect(self, handle):
        return handle

    def submit_prefill(self, layer_idx, y, selected, weights):
        return self.submit_issue(layer_idx, y, selected, weights)

    def submit_issue_fused(self, *a, **k):
        raise NotImplementedError("null host: EXL3_MOE_SPLIT_FUSED must be 0")

    def unregister(self):
        pass

    def shutdown(self):
        pass


def install():
    import qmesh.remote_host as rh
    rh.RemoteExpertHost = NullRemoteExpertHost
    # model.load() spawns the local CPU MoE worker whenever moe_cpu_split is set and pumps it on
    # every module commit. With the null remote there is no CPU worker: stub both (harness-only).
    from exllamav3.model.moe_cpu_host import MoeCpuHost
    MoeCpuHost._spawn = lambda self: None
    MoeCpuHost.commit_module = lambda self, key: None
    MoeCpuHost.shutdown = lambda self: None
    return NullRemoteExpertHost
