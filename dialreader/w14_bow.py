"""Week 14 - Image classification with Bag of Words (Stage 9): gauge type from SIFT 'visual words'."""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC


def sift_descriptors(face, size=256, max_features=400):
    grey = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    _, desc = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(grey, None)
    return desc if desc is not None else np.zeros((0, 128), np.float32)


class BagOfWords:
    """Vocabulary of visual words (k-means on SIFT descriptors), word histograms, and an SVM."""

    def __init__(self, n_words=120, seed=0):
        self.n_words = n_words
        self.kmeans = MiniBatchKMeans(n_clusters=n_words, random_state=seed, n_init=3, batch_size=2048)
        self.svm = SVC(kernel="linear", C=10.0)

    def histogram(self, desc):
        hist = np.zeros(self.n_words, np.float32)
        if len(desc):
            hist += np.bincount(self.kmeans.predict(desc.astype(np.float32)), minlength=self.n_words)
        # Square-root (Hellinger) scaling: stops a few very frequent words from dominating.
        return np.sqrt(hist / (hist.sum() + 1e-6))

    def fit(self, faces, labels):
        descs = [sift_descriptors(f) for f in faces]
        self.kmeans.fit(np.vstack([d for d in descs if len(d)]).astype(np.float32))
        self.svm.fit(np.array([self.histogram(d) for d in descs]), labels)
        return self

    def predict(self, faces):
        return self.svm.predict(np.array([self.histogram(sift_descriptors(f)) for f in faces]))
