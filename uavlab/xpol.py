"""Load an .xpol policy image on the host the same way PX4's ext_policy does, for offline testing.

Verifies header and CRC, maps the code into executable memory and calls xpol_step through ctypes.
"""
import ctypes
import mmap
import struct
import zlib
from pathlib import Path

import numpy as np

HEADER_FMT = "<IHHIIIIIHH16s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
XPOL_MAGIC = 0x4C4F5058
XPOL_NO_ENTRY = 0xFFFFFFFF

_libm = ctypes.CDLL("libm.so.6")
_F1 = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)
_F2 = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_float)


class MathTable(ctypes.Structure):
    _fields_ = [("sinf", ctypes.c_void_p), ("cosf", ctypes.c_void_p), ("sqrtf", ctypes.c_void_p),
                ("atan2f", ctypes.c_void_p), ("expf", ctypes.c_void_p)]


_MATH = MathTable(*(ctypes.cast(getattr(_libm, n), ctypes.c_void_p) for n in ("sinf", "cosf", "sqrtf", "atan2f", "expf")))
_STEP = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(MathTable),
                         ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
_INIT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(MathTable))


def verify(data: bytes) -> tuple:
    if len(data) < HEADER_SIZE:
        raise ValueError("file too small")
    hdr = list(struct.unpack_from(HEADER_FMT, data))
    magic, abi, hsize, size, crc = hdr[:5]
    if magic != XPOL_MAGIC or abi != 1 or hsize != HEADER_SIZE:
        raise ValueError("bad header")
    if size != len(data):
        raise ValueError(f"size mismatch: header {size}, file {len(data)}")
    zeroed = hdr.copy()
    zeroed[4] = 0
    calc = zlib.crc32(struct.pack(HEADER_FMT, *zeroed) + data[HEADER_SIZE:])
    if calc != crc:
        raise ValueError(f"CRC mismatch: header 0x{crc:08x}, computed 0x{calc:08x}")
    return tuple(hdr)


class Policy:
    def __init__(self, path: Path):
        data = Path(path).read_bytes()
        _, _, _, _, self.crc, init_off, step_off, state_size, _, _, name = verify(data)
        self.name = name.rstrip(b"\0").decode()
        code = data[HEADER_SIZE:]
        self._mem = mmap.mmap(-1, max(len(code), 1), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        self._mem.write(code)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mem))
        self._state = ctypes.create_string_buffer(max(state_size, 1))
        self._step = _STEP(base + step_off)
        if init_off != XPOL_NO_ENTRY and _INIT(base + init_off)(self._state, ctypes.byref(_MATH)) != 0:
            raise RuntimeError("policy init failed")
        self._x = (ctypes.c_float * 12)()
        self._u = (ctypes.c_float * 3)()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x = [p, v, a, j] (12,), returns snap (3,)."""
        self._x[:] = [float(v) for v in x]
        self._step(self._state, ctypes.byref(_MATH), self._x, self._u)
        return np.array(self._u[:])
