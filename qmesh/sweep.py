import socket, struct, pickle, time, json, sys
import torch

placement = json.load(open("/cold/placement.exl3.json"))
drop = int(sys.argv[1]) if len(sys.argv) > 1 else 0
s = socket.create_connection(("127.0.0.1", 29517))


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


def reqs():
    total = 0
    t0 = time.time()
    for layer in range(40):
        cold = placement["layers"][str(layer)]["warm"]
        if drop:
            cold = cold[:-drop]
        n = len(cold)
        for start in range(0, n, 6):
            ids = [min(start + k, n - 1) for k in range(6)]
            send({"cmd": "forward", "layer": layer,
                  "x": torch.randn(1, 5120, dtype=torch.half),
                  "ids": torch.tensor([ids], dtype=torch.long),
                  "weights": torch.full((1, 6), 1 / 6, dtype=torch.float)})
            recv()
            total += 6
    return total, time.time() - t0


# warm code path
before = stats()
total, dt = reqs()
mid = stats()
total2, dt2 = reqs()
after = stats()
print(json.dumps({
    "drop_per_layer": drop,
    "pass1_s": dt, "pass1_majflt": mid["majflt"] - before["majflt"],
    "pass2_s": dt2, "pass2_majflt": after["majflt"] - mid["majflt"],
    "allocated_gib": after["allocated"] / 2**30,
    "mem_free_gib": after["mem_free_kb"] / 1024 / 1024,
    "mem_available_gib": after["mem_available_kb"] / 1024 / 1024,
}), flush=True)
