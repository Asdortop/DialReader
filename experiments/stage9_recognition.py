"""Stage 9 (optional) - Recognise the gauge type, and find the gauge in a cluttered scene.

HOG + SVM detector, and three gauge-type classifiers
(HOG, LBP, Bag-of-Words on SIFT) compared on accuracy and confusion matrices.

Run:  python experiments/stage9_recognition.py
"""
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

from common import DATA, FIGS, RESULTS, SYNTH, load_synth, save_table
from dialreader import pipeline, viz, attributes as attrs, geometry as geo
from dialreader import hog, lbp, bow

FACE = int(1.06 * geo.RECT_RADIUS)


def dial_face(path):
    """The frontal dial face from Stages 1 to 3 (no scale is needed for these)."""
    img = attrs.read_image(path)
    R = pipeline.read_dial(img, {"vmin": 0.0, "vmax": 1.0}, keep_images=True)
    if "rectified" in R.images:
        c = geo.RECT_SIZE // 2
        return cv2.resize(R.images["rectified"][c - FACE:c + FACE, c - FACE:c + FACE], (256, 256))
    h, w = img.shape[:2]                      # dial not found: fall back to the centre of the photo
    s = min(h, w) // 2
    return cv2.resize(img[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s], (256, 256))


def faces_of(records):
    return Parallel(n_jobs=-1)(delayed(dial_face)(r["path"]) for r in records)


def truth_box(rec):
    cx, cy, r = float(rec["cx"]), float(rec["cy"]), float(rec["r_px"])
    return (cx - r, cy - r, cx + r, cy + r)


def crop(img, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    return img[y0:max(y0 + 8, y1), x0:max(x0 + 8, x1)]


def detector_patches(records, rng, negatives_per_image=6):
    pos, neg = [], []
    for rec in records:
        img = attrs.read_image(rec["path"])
        box = truth_box(rec)
        pos.append(crop(img, box))
        pos.append(cv2.flip(crop(img, box), 1))
        h, w = img.shape[:2]
        added = tries = 0
        while added < negatives_per_image and tries < 60:     # background windows that miss the gauge
            tries += 1
            size = int(rng.uniform(150, 420))
            x0, y0 = int(rng.uniform(0, max(1, w - size))), int(rng.uniform(0, max(1, h - size)))
            cand = (x0, y0, x0 + size, y0 + size)
            if hog.iou(cand, box) < 0.25:
                neg.append(crop(img, cand))
                added += 1
    return pos, neg


def evaluate_detector(clf, records):
    """Score every detection on the test photos as true or false positive (IoU >= 0.5)."""
    hits = []
    for rec in records:
        boxes, scores = hog.detect(attrs.read_image(rec["path"]), clf, threshold=-1.0)
        box, found = truth_box(rec), False
        for b, s in zip(boxes, scores):
            ok = (not found) and hog.iou(b, box) >= 0.5
            found = found or ok
            hits.append((s, ok))
    hits.sort(key=lambda t: -t[0])
    tp = np.cumsum([h for _, h in hits])
    fp = np.cumsum([not h for _, h in hits])
    recall, precision = tp / len(records), tp / np.maximum(tp + fp, 1)
    ap = float(np.trapezoid(precision, recall)) if len(recall) > 1 else 0.0
    scores = np.array([s for s, _ in hits])
    return recall, precision, ap, scores


if __name__ == "__main__":
    records = load_synth()
    # Stage 9 has its own split, because the main split holds two gauge types out entirely and a
    # classifier cannot name a type it has never seen: even readings train, odd readings test.
    train = [r for r in records if int(r["reading"]) % 2 == 0]
    test = [r for r in records if int(r["reading"]) % 2 == 1]
    stress = [r for r in attrs.load_dataset(DATA / "synthetic_stress") if int(r["reading"]) % 2 == 1]
    classes = sorted({r["gauge"] for r in records})
    y_train = [r["gauge"] for r in train]
    f_train, f_test, f_stress = faces_of(train), faces_of(test), faces_of(stress)

    # ---- three gauge-type classifiers
    models = {
        "HOG + linear SVM": (lambda: LinearSVC(C=1.0, max_iter=5000).fit([hog.hog(f) for f in f_train], y_train),
                             lambda m, fs: m.predict([hog.hog(f) for f in fs])),
        "LBP + SVM": (lambda: lbp.train(f_train, y_train), lambda m, fs: lbp.predict(m, fs)),
        "Bag of Words (SIFT) + SVM": (lambda: bow.BagOfWords().fit(f_train, y_train), lambda m, fs: m.predict(fs)),
    }
    rows, matrices = [], {}
    for name, (fit, predict) in models.items():
        model = fit()
        pred = predict(model, f_test)
        pred_stress = predict(model, f_stress)
        acc = float(np.mean(np.array(pred) == [r["gauge"] for r in test]))
        acc_s = float(np.mean(np.array(pred_stress) == [r["gauge"] for r in stress]))
        rows.append({"classifier": name, "train photos": len(train), "test photos": len(test),
                     "accuracy % (test photos)": 100 * acc, "stress photos": len(stress),
                     "accuracy % (stress photos)": 100 * acc_s})
        matrices[name] = confusion_matrix([r["gauge"] for r in stress], pred_stress, labels=classes)
        print(rows[-1])
    save_table(pd.DataFrame(rows).round(1), "stage9_classifiers")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    short = [c.replace("_", " ") for c in classes]
    for ax, (name, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > 0.55 * cm.max() else viz.INK)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(short if ax is axes[0] else [], fontsize=8)
        ax.set(title=name, xlabel="predicted type")
        ax.grid(False)
    axes[0].set_ylabel("true type")
    fig.suptitle("Stage 9: gauge-type confusion matrices on the stress photos (rows: true type)", x=0.01, ha="left",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "stage9_confusion.png")

    # ---- HOG + SVM gauge detector (sliding window), with one round of hard-negative mining
    rng = np.random.default_rng(0)
    pos, neg = detector_patches(train, rng)
    clf = hog.train(pos, neg)
    hard = []
    for rec in train[::3]:
        img = attrs.read_image(rec["path"])
        boxes, _ = hog.detect(img, clf, threshold=0.0, max_det=8)
        hard += [crop(img, b) for b in boxes if hog.iou(b, truth_box(rec)) < 0.25]
    clf = hog.train(pos, neg + hard)
    (RESULTS / "models").mkdir(exist_ok=True)
    with open(RESULTS / "models" / "hog_detector.pkl", "wb") as f:
        pickle.dump(clf, f)

    det_rows = []
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for (name, recs), colour in zip((("Test photos", test), ("Stress photos", stress)), viz.SERIES):
        recall, precision, ap, _ = evaluate_detector(clf, recs)
        top = [hog.detect(attrs.read_image(r["path"]), clf, threshold=-1.0, max_det=1) for r in recs]
        found = np.mean([bool(b) and hog.iou(b[0], truth_box(r)) >= 0.5 for (b, _), r in zip(top, recs)])
        det_rows.append({"data": name, "photos": len(recs), "average precision": ap,
                         "best box is the gauge % (IoU >= 0.5)": 100 * float(found),
                         "training positives": len(pos), "training negatives": len(neg) + len(hard)})
        ax.plot(100 * recall, 100 * precision, color=colour, label=f"{name} (AP {ap:.2f})")
        print(det_rows[-1])
    ax.set(xlabel="recall (%)", ylabel="precision (%)", xlim=(0, 101), ylim=(0, 101),
           title="Stage 9: HOG + SVM gauge detector")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "stage9_detector_pr.png")
    save_table(pd.DataFrame(det_rows).round(2), "stage9_detector")

    # ---- technique figures
    sample = attrs.read_image(test[3]["path"])
    boxes, scores = hog.detect(sample, clf, threshold=0.0, max_det=3)
    shown = sample.copy()
    for b, s in zip(boxes, scores):
        cv2.rectangle(shown, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), viz.AQUA, 3)
        cv2.putText(shown, f"{s:.2f}", (int(b[0]) + 6, int(b[1]) + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, viz.AQUA, 2)
    patch = cv2.resize(cv2.cvtColor(crop(sample, truth_box(test[3])), cv2.COLOR_BGR2GRAY), (128, 128))
    viz.show_images([patch, hog.hog_image(patch), shown],
                    ["Gauge patch", "Its HOG cells (9 orientations each)", "Sliding-window detections"],
                    suptitle="HOG + SVM gauge detector").savefig(FIGS / "hog_detector.png")

    face = cv2.cvtColor(f_test[0], cv2.COLOR_BGR2GRAY)
    codes = lbp.lbp_codes(cv2.resize(face, (128, 128)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    axes[0].imshow(face, cmap="gray"); axes[0].set_title("Rectified dial face"); axes[0].axis("off")
    axes[1].imshow(codes, cmap="gray"); axes[1].set_title("LBP code of every pixel"); axes[1].axis("off")
    hist = np.bincount(lbp.UNIFORM[codes].ravel(), minlength=lbp.N_BINS)
    axes[2].bar(range(lbp.N_BINS), hist / hist.sum(), color=viz.SERIES[0], width=0.8)
    axes[2].set(title="Uniform-LBP histogram (59 bins)", xlabel="pattern bin", ylabel="share of pixels")
    fig.suptitle("Local binary patterns", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "lbp_texture.png")

    model = bow.BagOfWords().fit(f_train, y_train)
    grey = cv2.cvtColor(f_test[0], cv2.COLOR_BGR2GRAY)
    kps = cv2.SIFT_create(nfeatures=400).detect(grey, None)
    drawn = cv2.drawKeypoints(f_test[0], kps, None, color=viz.ORANGE,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.5, 1.5]})
    axes[0].imshow(viz.rgb(drawn)); axes[0].set_title(f"SIFT keypoints ({len(kps)})"); axes[0].axis("off")
    for ax, idx, colour in ((axes[1], 0, viz.SERIES[0]), (axes[2], len(f_test) // 2, viz.SERIES[1])):
        ax.bar(range(model.n_words), model.histogram(bow.sift_descriptors(f_test[idx])), color=colour, width=0.9)
        ax.set(title=f"Visual-word histogram: {test[idx]['gauge'].replace('_', ' ')}", xlabel="visual word",
               ylabel="weight")
    fig.suptitle("Bag of Words on SIFT descriptors", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "bag_of_words.png")
