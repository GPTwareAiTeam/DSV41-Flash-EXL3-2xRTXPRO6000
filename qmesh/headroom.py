import socket, struct, pickle, time, json, sys
import torch

HOST = "127.0.0.1"
PORT = 29517
placement = json.load(open("/cold/placement.exl3.json"))
drop = int(sys.argv[1]) if len(sys.argv) > 1 else 0

s = socket.create_connection((HOST, PORT))


def send(o):
    b = pickle.dumps(o, protocol=pickle.HIGHEST_PROTOCOL)
    s.sendall(struct.pack("<Q", len(b)) + b)


def recv():
    h = s.recv(8)
    n = struct.unpack("<Q", h)[0]
    b = bytearray()
    while len(b) < n:
        b += s.recv(n - len(b))
    return pickle.loads(b)


def stats():
    send({"cmd": "stats"})
    return recv()

before = stats()
t0 = time.time()
total_assignments = 0
for layer in range(40):
    cold = placement["layers"][str(layer)]["warm"]
    if drop:
        cold = cold[:-drop]
    n = len(cold)
    ids = []
    for r in range(1):
        row = [(layer * 37 + k * 11) % n for k in range(6)]
        ids.append(row)
    x = torch.randn(1, 5120, dtype=torch.half)
    w = torch.full((1, 6), 1 / 6, dtype=torch.float)
    send({"cmd": "forward", "layer": layer, "x": x,
          "ids": torch.tensor(ids, dtype=torch.long), "weights": w})
    recv()
    total_assignments += 6
dt = time.time() - t0
after = stats()
print(json.dumps({
    "drop_per_layer": drop,
    "latency_s": dt,
    "per_layer_ms": 1000 * dt / 40,
    "assignments": total_assignments,
    "majflt_delta": after["majflt"] - before["majflt"],
    "minflt_delta": after["minflt"] - before["minflt"],
    "allocated_gib": after["allocated"] / 2**30,
    "reserved_gib": after["reserved"] / 2**30,
    "mem_available_gib": after["mem_available_kb"] / 1024 / 1024,
    "mem_free_gib": after["mem_free_kb"] / 1024 / 1024,
}), flush=True)
