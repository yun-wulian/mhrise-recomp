"""Summarize one face probe capture. Use --pixel X Y for a known dark point on the NPC."""
import argparse
from collections import Counter
from pathlib import Path
import struct

KINDS = ("RGB export only", "interpolated input", "texture sample", "division", "math function")


def unsigned_float(bits, mantissa_bits):
    exponent, fraction = divmod(bits, 1 << mantissa_bits)
    if exponent == 31:
        return float("nan") if fraction else float("inf")
    if exponent == 0:
        return fraction * 2.0 ** (-14 - mantissa_bits)
    return (1 + fraction / (1 << mantissa_bits)) * 2.0 ** (exponent - 15)


def marker(word):
    blue = unsigned_float(word >> 22, 5)
    if blue not in range(16, 16 + len(KINDS)):
        return None
    red = unsigned_float(word & 2047, 6)
    green = unsigned_float((word >> 11) & 2047, 6)
    if red not in range(1, 129) or green not in range(1, 129):
        return None
    result = int(green - 1) * 128 + int(red - 1)
    kind = int(blue) - 16
    if bool(result) != bool(kind):
        return None
    return result, kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--pixel", type=int, nargs=2, metavar=("X", "Y"))
    args = parser.parse_args()
    data = args.raw.read_bytes()
    width, height, fmt = struct.unpack_from("<3I", data)
    if fmt != 122 or len(data) != 12 + width * height * 4:
        parser.error("Expected an R11G11B10F face probe dump")
    if args.pixel:
        x, y = args.pixel
        if not (0 <= x < width and 0 <= y < height):
            parser.error(f"Pixel outside {width} x {height}")
        words = [struct.unpack_from("<I", data, 12 + 4 * (width * y + x))]
    else:
        words = struct.iter_unpack("<I", memoryview(data)[12:])
    counts = Counter(hit for (word,) in words if (hit := marker(word)) is not None)
    for (result, kind), count in counts.most_common():
        print(f"{count:7d} pixels  %{result:<6d}  {KINDS[kind]}")
    if not counts:
        print("No marker in selection")


if __name__ == "__main__":
    main()
