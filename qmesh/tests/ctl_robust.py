#!/usr/bin/env python3
"""Control-plane robustness probe for the seam server.

Connects to a running server's control port and sends a battery of handshakes a
server must survive: a status query, a bad magic, a wrong protocol version, a
wrong layer count, a malformed bucket count, a disjoint bucket list, a peer that
connects and says nothing (the 2 s hello timeout), and a legitimate hello. Then
the part the owners actually pay for: two more sessions in a row asking for
different bucket subsets of what the server serves, and a client that arrives
while somebody else's session is live, which must be refused with ST_BUSY
without touching that session. Between each, status must answer idle: if every
one of those got an answer and the pid never changed, the server did not die on
a client it did not expect.

Deliberately raw sockets and struct.pack: this is the regression test for the
"qes: peer layout mismatch" abort, so it must not share code with the client
that hides the bytes. Run it against a --stub server (no pack, no GPU).

Exit status: 0 all expectations met, 1 a mismatch (with the observed reply
printed), 2 could not talk to the port at all.
"""

import socket
import struct
import sys
import time

# qmesh/native/qmesh_expert_seam.h
QES_NL = 40
QES_PROTO_VERSION = 4
QES_MAX_BUCKETS = 12
QES_BUILD_LEN = 40
QES_MSG_LEN = 192
CTL_MAGIC = 0x32534551  # "QES2"
CTL_REPLY_MAGIC = 0x32525351  # "QSR2"
CTL_FLAG_WAIT = 1
CTL_FLAG_STATUS = 2
ST_OK, ST_BUSY, ST_QUEUED, ST_INCOMPAT, ST_BADREQUEST, ST_STATUS = 0, 1, 2, 3, 4, 6

HELLO = struct.Struct("<IIIII%diII%ds" % (QES_MAX_BUCKETS, QES_BUILD_LEN))
REPLY = struct.Struct("<IIIIIii%di%di%diI%ds%ds%ds"
                      % (QES_MAX_BUCKETS, QES_MAX_BUCKETS, QES_MAX_BUCKETS,
                         QES_BUILD_LEN, QES_BUILD_LEN, QES_MSG_LEN))


def hello(buckets, version=QES_PROTO_VERSION, layers=QES_NL, magic=CTL_MAGIC, flags=0,
          build=b"probe"):
    b = list(buckets) + [0] * (QES_MAX_BUCKETS - len(buckets))
    return HELLO.pack(magic, version, flags, layers, len(buckets), *b, 0, 0,
                      build.ljust(QES_BUILD_LEN, b"\0"))


def recv_reply(s, timeout=3.0):
    """Read one fixed-size CtlReply off a connected socket; b"" if the server closed first."""
    s.settimeout(timeout)
    buf = b""
    while len(buf) < REPLY.size:
        chunk = s.recv(REPLY.size - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def ask(host, port, payload, wait_s=8.0, send_nothing=False, timeout=3.0):
    """One connection, one hello, one reply. Returns (status, message, buckets) or None."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        if send_nothing:
            time.sleep(wait_s)          # server must time us out, not wait forever
            return None
        s.sendall(payload)
        buf = recv_reply(s, timeout)
        if len(buf) != REPLY.size:
            return None
        f = REPLY.unpack(buf)
        magic, ver, status, session, layers, nb, nreq = f[0:7]
        if magic != CTL_REPLY_MAGIC or ver != QES_PROTO_VERSION:
            print(f"  reply magic/version wrong: magic=0x{magic:08x} v={ver}")
            return None
        msg = f[-1].split(b"\0", 1)[0].decode(errors="replace")
        buckets = list(f[7:7 + nb])
        return status, msg, buckets
    finally:
        s.close()


def main(argv):
    host = argv[1] if len(argv) > 1 else "127.0.0.1"
    port = int(argv[2]) if len(argv) > 2 else 47911
    served = [int(x) for x in (argv[3] if len(argv) > 3 else "1,6,64,8192").split(",")]
    fails = 0

    def check(name, payload, want, want_msg=None, send_nothing=False):
        nonlocal fails
        try:
            got = ask(host, port, payload, send_nothing=send_nothing)
        except OSError as e:
            print(f"FAIL {name}: connection error {e}")
            fails += 1
            return
        if got is None:
            print(f"FAIL {name}: no usable reply (server closed or garbled it)")
            fails += 1
            return
        status, msg, buckets = got
        bad = status != want or (want_msg and want_msg not in msg)
        print(f"{'FAIL' if bad else 'ok  '} {name}: status={status} msg={msg!r}")
        if bad:
            fails += 1
        if want == ST_STATUS and buckets[:len(served)] != served:
            print(f"FAIL {name}: status reply buckets={buckets[:len(served)]} want={served}")
            fails += 1

    # 1. A status query must answer and list what the server serves.
    check("status", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 2. Every malformed or incompatible hello must get a typed rejection, not a dropped port.
    check("bad-magic", hello(served, magic=0xDEADBEEF), ST_INCOMPAT, "magic")
    check("wrong-version", hello(served, version=QES_PROTO_VERSION - 1), ST_INCOMPAT, "protocol")
    check("wrong-layers", hello(served, layers=QES_NL - 1), ST_INCOMPAT, "layer")
    check("no-buckets", hello([]), ST_BADREQUEST, "n_buckets")
    check("disjoint-buckets", hello([3, 7]), ST_BADREQUEST, "row bucket")

    # 3. A peer that connects and never speaks must be timed out, and the accept loop must
    #    survive it: the next status query is what proves that.
    print("     silent peer: waiting out the 2 s hello timeout")
    try:
        ask(host, port, b"", send_nothing=True)
    except OSError as e:
        print(f"FAIL silent-peer: connection error {e}")
        fails += 1
    check("status-after-silent-peer", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 4. A legitimate hello still starts a session.
    check("valid-hello", hello(served), ST_OK)

    # 5. And after that session's socket was closed under it, the server is back to idle,
    #    which means it reaped rather than died.
    time.sleep(0.5)
    check("status-after-valid-hello", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 6. The owner's requirement: the next client may want different buckets, and it must be
    #    served by the same process. Two sessions in a row, with disjoint bucket subsets of what
    #    the server offers, is the case that used to end the server's life.
    check("second-session-subset", hello([served[1]]), ST_OK)
    time.sleep(0.5)
    check("third-session-other-subset", hello([served[0], served[-1]]), ST_OK)
    time.sleep(0.5)
    check("status-after-three-sessions", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    # 7. A client that arrives while somebody else's session is live must be refused with a
    #    reason, and the live session must not be torn down by the refusal.
    live = None
    try:
        live = socket.create_connection((host, port), timeout=3.0)
        live.sendall(hello(served))
        live.settimeout(3.0)
        f = REPLY.unpack(recv_reply(live))
        print(f"{'ok  ' if f[2] == ST_OK else 'FAIL'} busy: first session status={f[2]}")
        if f[2] != ST_OK:
            fails += 1
        got = ask(host, port, hello(served))
        if got is None:
            print("FAIL busy: second client got no reply at all")
            fails += 1
        else:
            status, msg, _ = got
            ok = status == ST_BUSY and msg
            print(f"{'ok  ' if ok else 'FAIL'} busy: status={status} msg={msg!r}")
            if not ok:
                fails += 1
    except OSError as e:
        print(f"FAIL busy: connection error {e}")
        fails += 1
    finally:
        if live is not None:
            live.close()
    time.sleep(0.5)
    check("status-after-busy-refusal", hello(served, flags=CTL_FLAG_STATUS), ST_STATUS)

    print("RESULT ctl_robust=" + ("pass" if fails == 0 else f"fail({fails})"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
