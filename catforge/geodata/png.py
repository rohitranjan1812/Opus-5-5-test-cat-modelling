"""Minimal PNG decoder (8-bit, non-interlaced; grey/RGB/RGBA/palette) — no imaging dependency.

Scanline filters (None/Sub/Up/Average/Paeth, RFC 2083 §6) are undone in a numba kernel.
"""

from __future__ import annotations

import struct
import zlib

import numba as nb
import numpy as np

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


@nb.njit(cache=True)
def _unfilter(raw, h, stride, bpp):
    out = np.zeros((h, stride), np.uint8)
    p = 0
    for y in range(h):
        f = raw[p]
        p += 1
        for x in range(stride):
            a = np.int32(out[y, x - bpp]) if x >= bpp else 0
            b = np.int32(out[y - 1, x]) if y > 0 else 0
            c = np.int32(out[y - 1, x - bpp]) if (y > 0 and x >= bpp) else 0
            v = np.int32(raw[p + x])
            if f == 0:
                r = v
            elif f == 1:
                r = v + a
            elif f == 2:
                r = v + b
            elif f == 3:
                r = v + (a + b) // 2
            else:
                pa = abs(b - c)
                pb = abs(a - c)
                pc = abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                r = v + pred
            out[y, x] = r & 255
        p += stride
    return out


def decode_png(data: bytes) -> np.ndarray:
    """Decode to an (h, w, channels) uint8 array (palette images are expanded to RGB)."""
    if data[:8] != SIGNATURE:
        raise ValueError("not a PNG")
    pos, idat, palette, hdr = 8, [], None, None
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"PLTE":
            palette = np.frombuffer(chunk, np.uint8).reshape(-1, 3)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"IEND":
            break
    if hdr is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, ctype, _, _, interlace = hdr
    if depth != 8 or interlace or ctype not in _CHANNELS:
        raise ValueError(f"unsupported PNG (bit depth {depth}, colour type {ctype}, interlace {interlace})")
    ch = _CHANNELS[ctype]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    img = _unfilter(raw, h, w * ch, ch).reshape(h, w, ch)
    if ctype == 3:
        if palette is None:
            raise ValueError("palette PNG without PLTE")
        img = palette[img[..., 0]]
    return img
