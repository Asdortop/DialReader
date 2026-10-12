"""Week 13 - Local binary patterns (Stage 9): classify the gauge type from the texture of its face."""
import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _uniform_lookup():
    """Map each 8-bit pattern to a bin: the 58 'uniform' patterns (at most two 0/1 changes
    round the circle) get a bin each, every other pattern shares bin 58."""
    table = np.full(256, 58, np.uint8)
    nxt = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        if sum(bits[i] != bits[(i + 1) % 8] for i in range(8)) <= 2:
            table[code] = nxt
            nxt += 1
    return table


UNIFORM = _uniform_lookup()
N_BINS = 59


def lbp_codes(grey, margin=0):
    """8-neighbour LBP code of every interior pixel: one bit per neighbour that is >= the centre.

    `margin` asks a neighbour to exceed the centre by that many grey levels, so that sensor
    noise on a flat dial face does not produce random codes.
    """
    g = grey.astype(np.int16)
    centre = g[1:-1, 1:-1] + margin
    code = np.zeros(centre.shape, np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((neighbour >= centre).astype(np.uint8) << bit)
    return code


def lbp_features(face, grid=5, size=160):
    """Uniform-LBP histograms over a grid of cells, concatenated: a texture signature that
    also keeps a coarse layout of where the print is."""
    grey = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    codes = UNIFORM[lbp_codes(grey, margin=3)]
    step = codes.shape[0] // grid
    feats = []
    for r in range(grid):
        for c in range(grid):
            cell = codes[r * step:(r + 1) * step, c * step:(c + 1) * step]
            hist = np.bincount(cell.ravel(), minlength=N_BINS).astype(np.float32)
            feats.append(np.sqrt(hist / (hist.sum() + 1e-6)))      # square-root scaling, as for the word histograms
    return np.concatenate(feats)


def train(faces, labels):
    X = np.array([lbp_features(f) for f in faces])
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.05)).fit(X, labels)


def predict(clf, faces):
    return clf.predict(np.array([lbp_features(f) for f in faces]))
