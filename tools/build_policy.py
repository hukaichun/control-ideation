"""Compile a C policy into a position-independent .xpol image for the PX4 ext_policy module.

    uv run tools/build_policy.py policies/circle.c -o build/policies/circle.xpol

The policy must define `void xpol_step(void*, const xpol_math_t*, const float*, float*)` and may define
`int xpol_init(void*, const xpol_math_t*)` and `const uint32_t xpol_state_size`. It must not call libc or
use mutable globals; the build fails if the object has external symbols, absolute relocations or data.
Target is the host (x86_64) for SITL.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ABI_DIR = Path(__file__).resolve().parents[1] / "PX4-Autopilot/src/modules/ext_policy"

XPOL_MAGIC = 0x4C4F5058
XPOL_ABI_VERSION = 1
XPOL_NO_ENTRY = 0xFFFFFFFF
HEADER_FMT = "<IHHIIIIIHH16s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 48

CFLAGS = [
    "-O2", "-std=c11", "-Wall", "-Werror",
    "-fPIE", "-fvisibility=hidden", "-ffreestanding", "-fno-builtin", "-fno-math-errno",
    "-fno-stack-protector", "-fno-asynchronous-unwind-tables", "-fno-unwind-tables",
    "-fcf-protection=none", "-fno-jump-tables", "-fno-plt",
]
# PC-relative only: the image runs at whatever address the host maps it to
ALLOWED_RELOCS = {"R_X86_64_PC32", "R_X86_64_PLT32"}

LINKER_SCRIPT = """
SECTIONS {
  . = 0;
  .text   : { *(.text*) }
  .rodata : { *(.rodata*) }
  .data   : { *(.data*) *(.bss*) *(COMMON) }
  /DISCARD/ : { *(.eh_frame*) *(.comment) *(.note*) }
}
"""


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(map(str, cmd))}\n{r.stderr}")
    return r.stdout


def section_size(elf, name):
    for line in run(["readelf", "-SW", elf]).splitlines():
        parts = line.replace("[ ", "[").split()
        if len(parts) > 6 and parts[1] == name:
            return int(parts[5], 16)
    return 0


def build(src: Path, out: Path, name: str):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        obj, elf, code_bin, lds = tmp / "p.o", tmp / "p.elf", tmp / "code.bin", tmp / "p.ld"

        run(["gcc", *CFLAGS, f"-I{ABI_DIR}", "-c", src, "-o", obj])

        undefined = run(["nm", "-u", obj]).split()
        if undefined:
            sys.exit(f"policy calls external symbols {undefined[1::2] or undefined}; use the xpol_math_t table")

        relocs = {parts[2] for parts in map(str.split, run(["readelf", "-rW", obj]).splitlines())
                  if len(parts) > 2 and parts[2].startswith("R_")}
        bad = sorted(relocs - ALLOWED_RELOCS)
        if bad:
            sys.exit(f"non PC-relative relocations {bad}: code would depend on its load address")

        lds.write_text(LINKER_SCRIPT)
        run(["ld", "-T", lds, "-o", elf, obj])
        if section_size(elf, ".data"):
            sys.exit("policy has mutable global data; keep state in the host-provided state memory")

        run(["objcopy", "-O", "binary", "-j", ".text", "-j", ".rodata", elf, code_bin])
        code = code_bin.read_bytes()

        syms = {}
        for line in run(["nm", elf]).splitlines():
            addr, _, sym = line.split()
            syms[sym] = int(addr, 16)

    if "xpol_step" not in syms:
        sys.exit("xpol_step not defined")
    init_off = syms.get("xpol_init", XPOL_NO_ENTRY)
    state_size = 0
    if "xpol_state_size" in syms:
        state_size = struct.unpack_from("<I", code, syms["xpol_state_size"])[0]

    fields = [XPOL_MAGIC, XPOL_ABI_VERSION, HEADER_SIZE, HEADER_SIZE + len(code), 0,
              init_off, syms["xpol_step"], state_size, 12, 3, name.encode()[:16]]
    crc = zlib.crc32(struct.pack(HEADER_FMT, *fields) + code)
    fields[4] = crc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(struct.pack(HEADER_FMT, *fields) + code)
    print(f"{out}: '{name}' {len(code)} bytes code, step @0x{syms['xpol_step']:x}, crc 0x{crc:08x}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--name")
    a = ap.parse_args()
    build(a.src, a.out or Path("build/policies") / f"{a.src.stem}.xpol", a.name or a.src.stem)
