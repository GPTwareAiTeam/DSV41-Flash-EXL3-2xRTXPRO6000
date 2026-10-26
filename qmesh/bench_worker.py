import socket, struct, pickle, time
import torch

HOST, PORT = "10.0.100.2", 29517


def recv_msg(conn):
    h = conn.recv(8)
    n = struct.unpack("<Q", h)[0]
    b = bytearray()
    while len(b) < n:
        b += conn.recv(n - len(b))
    return pickle.loads(b)


def send_msg(conn, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("<Q", len(b)) + b)


s = socket.create_connection((HOST, PORT))
for rows in (1, 6, 48):
    x = torch.randn(rows, 5120, dtype=torch.half)
    ids = torch.arange(6, dtype=torch.long).repeat(rows, 1)
    w = torch.full((rows, 6), 1 / 6, dtype=torch.float)
    # warm
    send_msg(s, {"cmd": "forward", "layer": 0, "x": x, "ids": ids, "weights": w})
    recv_msg(s)
    t = time.time()
    n = 3
    for _ in range(n):
        send_msg(s, {"cmd": "forward", "layer": 0, "x": x, "ids": ids, "weights": w})
        out = recv_msg(s)
    dt = (time.time() - t) / n
    print(f"rows={rows} layer_ms={dt*1000:.1f} out_shape={tuple(out['out'].shape)}", flush=True)
s.close()
