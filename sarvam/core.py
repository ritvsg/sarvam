"""State and helpers shared by the interpreter and song endpoints: the Sarvam client, rooms, relay."""
import json
import time

from fastapi import HTTPException, WebSocket

from sarvam_client import SarvamError, get_client

client = get_client()
rooms: dict[str, dict[str, dict]] = {}  # room -> role -> {"ws": WebSocket, "lang": str | None}


class Run:
    """Timings and warnings for one request."""

    def __init__(self) -> None:
        self.timings: dict[str, int] = {}
        self.warnings: list[str] = []
        self.started = time.perf_counter()

    async def timed(self, name: str, awaitable):
        t = time.perf_counter()
        try:
            return await awaitable
        finally:
            self.timings[name] = self.timings.get(name, 0) + round((time.perf_counter() - t) * 1000)

    def finish(self) -> dict:
        self.timings["total"] = round((time.perf_counter() - self.started) * 1000)
        return self.timings


def other(role: str) -> str:
    return "B" if role == "A" else "A"


def peer_of(room: str, role: str) -> dict | None:
    return rooms.get(room, {}).get(other(role)) if room else None


async def send(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass  # the peer went away; its own disconnect handler cleans up


async def announce(room: str) -> None:
    peers = rooms.get(room, {})
    for role, me in list(peers.items()):
        partner = peers.get(other(role))
        await send(me["ws"], {"type": "peer", "connected": partner is not None,
                              "lang": partner["lang"] if partner else None})


async def skip_peer(peer: dict | None, seq: int, sid: str) -> None:
    """Tell the partner's playback queue not to wait for a segment that failed."""
    if peer:
        await send(peer["ws"], {"type": "skip", "seq": seq, "sid": sid})


def http_error(e: SarvamError) -> HTTPException:
    return HTTPException(502, detail={"stage": e.stage, "status": e.status, "body": e.body[:300]})


async def relay(peer: dict | None, result: dict) -> dict:
    """Send the result (with audio) to the partner if there is one, else hand it back to the caller."""
    if peer:
        await send(peer["ws"], {**result, "type": "utterance"})
        return {**result, "audio": None, "delivered": True}
    return {**result, "delivered": False}
