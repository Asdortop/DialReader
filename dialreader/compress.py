"""Stage 11 compression for upload.

A JPEG-style codec written out: 8x8 DCT, quantisation, zigzag scan, run-length
coding of the AC coefficients and Huffman code lengths for the size. It is
compared with the library JPEG encoder to see how small a dial photo can get
before the reading degrades.
"""
import heapq

import cv2
import numpy as np

from . import matrix as mx

# Standard JPEG quantisation tables (Annex K).
Q_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], np.float64)

C8 = mx.dct_matrix(8)


def zigzag_order(n=8):
    """Indices that read an n x n block along anti-diagonals, low frequencies first."""
    order = sorted(((r, c) for r in range(n) for c in range(n)),
                   key=lambda rc: (rc[0] + rc[1], rc[0] if (rc[0] + rc[1]) % 2 else rc[1]))
    return np.array([r * n + c for r, c in order])


ZIGZAG = zigzag_order()


def quality_table(base, quality):
    """Scale a quantisation table by the JPEG quality factor (1 = coarsest, 100 = finest)."""
    quality = int(np.clip(quality, 1, 100))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((base * scale + 50.0) / 100.0), 1, 255)


def to_blocks(channel):
    """Split a channel into 8x8 blocks (edge-padded to a multiple of 8): shape (rows, cols, 8, 8)."""
    h, w = channel.shape
    ph, pw = (-h) % 8, (-w) % 8
    padded = np.pad(channel.astype(np.float64), ((0, ph), (0, pw)), mode="edge")
    H, W = padded.shape
    return padded.reshape(H // 8, 8, W // 8, 8).swapaxes(1, 2)


def from_blocks(blocks, shape):
    rows, cols = blocks.shape[:2]
    return blocks.swapaxes(1, 2).reshape(rows * 8, cols * 8)[:shape[0], :shape[1]]


def dct2(blocks):
    """2D DCT of every block: C B C^T."""
    return np.einsum("ij,rcjk,lk->rcil", C8, blocks, C8)


def idct2(coefs):
    return np.einsum("ji,rcjk,kl->rcil", C8, coefs, C8)


def rle_encode(zz):
    """Run-length code one zigzag-ordered block's AC coefficients as (zero run, value) pairs.

    (0, 0) marks end of block: everything after it is zero.
    """
    pairs, run = [], 0
    last = np.nonzero(zz[1:])[0]
    end = last[-1] + 1 if len(last) else 0
    for v in zz[1:end + 1]:
        if v == 0:
            run += 1
        else:
            pairs.append((run, int(v)))
            run = 0
    pairs.append((0, 0))
    return pairs


def rle_decode(dc, pairs, n=64):
    zz = np.zeros(n, np.int64)
    zz[0] = dc
    i = 1
    for run, v in pairs:
        if (run, v) == (0, 0):
            break
        i += run
        zz[i] = v
        i += 1
    return zz


def huffman_lengths(freqs):
    """Code length in bits for each symbol of a frequency table (Huffman's algorithm)."""
    if len(freqs) == 1:
        return {next(iter(freqs)): 1}
    heap = [(f, i, [s]) for i, (s, f) in enumerate(freqs.items())]
    heapq.heapify(heap)
    lengths = dict.fromkeys(freqs, 0)
    counter = len(heap)
    while len(heap) > 1:
        f1, _, s1 = heapq.heappop(heap)
        f2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (f1 + f2, counter, s1 + s2))
        counter += 1
    return lengths


def _size_bits(v):
    """JPEG 'size category': the number of bits needed for the magnitude of v."""
    return abs(int(v)).bit_length()


def coded_bits(quantised):
    """Size in bits of one channel's quantised blocks: DPCM for DC, run-length pairs for AC,
    Huffman-coded symbols plus the raw magnitude bits (as in baseline JPEG)."""
    flat = quantised.reshape(-1, 64)[:, ZIGZAG].astype(np.int64)
    dc = np.diff(flat[:, 0], prepend=0)
    dc_sym, ac_sym, extra = {}, {}, 0
    for d in dc:
        s = _size_bits(d)
        dc_sym[s] = dc_sym.get(s, 0) + 1
        extra += s
    for block in flat:
        for run, v in rle_encode(block):
            while run > 15:                       # JPEG's ZRL symbol: sixteen zeros
                ac_sym[(15, 0)] = ac_sym.get((15, 0), 0) + 1
                run -= 16
            s = _size_bits(v)
            ac_sym[(run, s)] = ac_sym.get((run, s), 0) + 1
            extra += s
    bits = extra
    for table in (dc_sym, ac_sym):
        lengths = huffman_lengths(table)
        bits += sum(table[s] * lengths[s] for s in table)
    return bits


def compress(bgr, quality):
    """Encode and decode an image with the hand-written codec.

    Colour is converted to YCrCb and the three channels are coded at full
    resolution (no chroma subsampling). Returns (decoded BGR image, size in bytes).
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    out = np.zeros_like(ycc)
    bits = 0
    for c in range(3):
        table = quality_table(Q_LUMA if c == 0 else Q_CHROMA, quality)
        quantised = np.round(dct2(to_blocks(ycc[..., c]) - 128.0) / table)
        bits += coded_bits(quantised)
        out[..., c] = np.clip(from_blocks(idct2(quantised * table) + 128.0, ycc.shape[:2]), 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR), int(np.ceil(bits / 8.0)) + 600   # ~600 bytes of headers


def library_jpeg(bgr, quality):
    """The same with OpenCV's JPEG encoder. Returns (decoded BGR image, size in bytes)."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), len(buf)
