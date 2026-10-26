import socket, struct, pickle, time
import torch

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

x = torch.randn(1, 5120, dtype=torch.half)
ids = torch.arange(6, dtype=torch.long).view(1, 6)
w = torch.full((1, 6), 1/6, dtype=torch.float)
t = time.time(); send({"cmd":"forward","layer":0,"x":x,"ids":ids,"weights":w}); o=recv(); print("one_ms", (time.time()-t)*1000, o["out"].shape, flush=True)
s.close()
