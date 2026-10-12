"""Stage 11 (optional) - How far can a dial photo be compressed before the reading degrades?

Week 7: the hand-written DCT codec against the library JPEG encoder, with reading
error against quality factor and PSNR against compression ratio.

Run:  python experiments/stage11_compression.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import FIGS, evaluate_image, load_synth, save_table
from dialreader import viz, w02_attributes as attrs, w05_restore as restore, w07_compress as codec

QUALITIES = [90, 70, 50, 30, 20, 10, 5, 2]


def one(rec, quality):
    img = attrs.read_image(rec["path"])
    raw_bytes = img.size
    out = {"file": rec["file"], "quality": quality}
    for name, fn in (("own", codec.compress), ("library", codec.library_jpeg)):
        decoded, n_bytes = fn(img, quality)
        row = evaluate_image(decoded, rec)
        out.update({f"{name}_ratio": raw_bytes / n_bytes, f"{name}_psnr": restore.psnr(decoded, img),
                    f"{name}_correct": bool(row["within_division"]) if row["error_fs"] is not None else False,
                    f"{name}_reliable": row["status"] == "reliable",
                    f"{name}_err": row["error_fs"]})
    return out


if __name__ == "__main__":
    recs = load_synth(split="test", subset="base")[::2]
    rows = Parallel(n_jobs=-1)(delayed(one)(r, q) for r in recs for q in QUALITIES)
    df = pd.DataFrame(rows)
    table = []
    for q in QUALITIES:
        part = df[df["quality"] == q]
        entry = {"quality factor": q, "photos": len(part)}
        for name, label in (("own", "own codec"), ("library", "library JPEG")):
            entry.update({f"{label}: compression ratio": part[f"{name}_ratio"].mean(),
                          f"{label}: PSNR (dB)": part[f"{name}_psnr"].mean(),
                          f"{label}: read within 1 division %": 100 * part[f"{name}_correct"].mean(),
                          f"{label}: mean error (% FS)": part[f"{name}_err"].mean()})
        table.append(entry)
    out = save_table(pd.DataFrame(table).round(2), "stage11_compression")
    print(out.to_string(index=False))
    ok = out[out["library JPEG: read within 1 division %"] >= 95.0]
    lowest = int(ok["quality factor"].min()) if len(ok) else None
    print("lowest library-JPEG quality with >= 95 % of readings within one division:", lowest)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))
    for label, colour in (("own codec", viz.SERIES[0]), ("library JPEG", viz.SERIES[1])):
        axes[0].plot(out["quality factor"], out[f"{label}: read within 1 division %"], "-o", color=colour, label=label)
        axes[1].plot(out[f"{label}: compression ratio"], out[f"{label}: PSNR (dB)"], "-o", color=colour, label=label)
        axes[2].plot(out["quality factor"], out[f"{label}: compression ratio"], "-o", color=colour, label=label)
    axes[0].set(title="Readings that stay within one division", xlabel="JPEG quality factor", ylabel="% of photos",
                ylim=(-4, 104))
    axes[0].invert_xaxis()
    axes[1].set(title="PSNR against compression ratio", xlabel="compression ratio (raw bytes / coded bytes)",
                ylabel="PSNR (dB)")
    axes[2].set(title="Compression ratio against quality", xlabel="JPEG quality factor", ylabel="compression ratio")
    axes[2].invert_xaxis()
    axes[0].legend(loc="lower left")
    fig.suptitle("Stage 11: compression for upload", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "stage11_compression.png")

    # ---- week 7 figure: what the codec does to one 8x8 block and to the whole photo
    img = attrs.read_image(recs[0]["path"])
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    y, x = h // 2 - 40, w // 2 + 56
    block = grey[y:y + 8, x:x + 8].astype(np.float64)
    coef = codec.C8 @ (block - 128) @ codec.C8.T
    table10 = codec.quality_table(codec.Q_LUMA, 10)
    quant = np.round(coef / table10)
    back = np.clip(codec.C8.T @ (quant * table10) @ codec.C8 + 128, 0, 255)
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.1))
    for ax, data, title, cmap in ((axes[0], block, "8x8 block", "gray"),
                                  (axes[1], np.log1p(np.abs(coef)), "DCT coefficients (log)", "Blues"),
                                  (axes[2], np.abs(quant), "After quantisation (q = 10)", "Blues"),
                                  (axes[3], back, "Decoded block", "gray")):
        ax.imshow(data, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=255 if cmap == "gray" else None)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax, q in ((axes[4], 10), (axes[5], 2)):
        dec, n = codec.compress(img, q)
        c = dec[h // 2 - 150:h // 2 + 150, w // 2 - 150:w // 2 + 150]
        ax.imshow(viz.rgb(c)); ax.axis("off")
        ax.set_title(f"Own codec, quality {q}: {img.size / n:.0f}:1", fontsize=10)
    nz = int(np.count_nonzero(quant))
    fig.suptitle(f"Week 7 - Image compression: 8x8 DCT, quantisation ({nz} of 64 coefficients survive at "
                 "quality 10), run-length coding", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "week07_compression.png")
