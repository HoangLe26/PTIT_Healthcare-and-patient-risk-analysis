"""Batched exact Euclidean search for the group's KNN grid."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

K_VALUES = (5, 15, 31)
WEIGHTS = ("uniform", "distance")
BATCH_SIZE = 192
BLAS_THREADS = 2
MISSING_CATEGORY = "__missing__"


class EuclideanNeighbors:
    def __init__(self, train, numeric, categorical):
        self.train = train.sort_values("row_id")
        self.y = self.train["y"].to_numpy()
        self.numeric = numeric
        self.categorical = categorical
        self.scaler = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
        self.train_numeric = self.scaler.fit_transform(self.train[numeric])
        categories = self.categories(self.train)
        # Reserve a column for missingness even when this fold has no missing value.
        levels = [sorted(set(categories[col]) | {MISSING_CATEGORY}) for col in categorical]
        self.encoder = OneHotEncoder(categories=levels, handle_unknown="ignore")
        encoded = self.encoder.fit_transform(categories)
        self.train_cat_counts = np.diff(encoded.indptr)
        # Frequent one-hot columns use BLAS. The remaining columns use a sparse
        # dot product, which avoids multiplying thousands of mostly-zero columns.
        self.frequent = encoded.getnnz(axis=0) >= 0.10 * len(self.train)
        self.train_frequent_t = np.ascontiguousarray(encoded[:, self.frequent].toarray().T)
        self.train_rare_t = encoded[:, ~self.frequent].T.tocsr()

    def categories(self, frame):
        return frame[self.categorical].astype("string").fillna(MISSING_CATEGORY)

    def neighbors(self, query, batch_size=BATCH_SIZE):
        """Compute the nearest 31 once, ordered by distance and then training row ID."""
        numeric = self.scaler.transform(query[self.numeric])
        encoded = self.encoder.transform(self.categories(query))
        frequent = encoded[:, self.frequent].toarray()
        rare = encoded[:, ~self.frequent].tocsr()
        cat_counts = np.diff(encoded.indptr)
        max_k = max(K_VALUES)
        indices = np.empty((len(query), max_k), dtype=np.intp)
        distances = np.empty((len(query), max_k), dtype=float)

        with threadpool_limits(limits=BLAS_THREADS, user_api="blas"):
            for start in range(0, len(query), batch_size):
                block = slice(start, start + batch_size)
                squared = cdist(numeric[block], self.train_numeric, metric="sqeuclidean")
                # For binary one-hot vectors, ||a-b||² = nnz(a) + nnz(b) - 2 a·b.
                # Unknown levels have all zeros within their feature block.
                matches = frequent[block] @ self.train_frequent_t
                if self.train_rare_t.nnz:
                    matches += (rare[block] @ self.train_rare_t).toarray()
                matches *= -2
                matches += cat_counts[block, None]
                matches += self.train_cat_counts
                squared += matches
                for i, row in enumerate(squared, start=start):
                    prefix = np.argpartition(row, max_k - 1)[:max_k]
                    cutoff = row[prefix[-1]]
                    better = prefix[row[prefix] < cutoff]
                    tied = np.flatnonzero(row == cutoff)[:max_k - len(better)]
                    nearest = np.concatenate((better, tied))
                    nearest = nearest[np.lexsort((nearest, row[nearest]))]
                    indices[i] = nearest
                    distances[i] = np.sqrt(row[nearest])
        return distances, indices

    def mean_squared_components(self, sample, indices):
        """Mean contribution of each original feature to the selected distances."""
        numeric = self.scaler.transform(sample[self.numeric])[0]
        numeric_components = np.square(self.train_numeric[indices] - numeric).mean(axis=0)
        encoded_sample = self.encoder.transform(self.categories(sample)).toarray()[0]
        encoded_train = self.encoder.transform(self.categories(self.train.iloc[indices])).toarray()
        differences = np.square(encoded_train - encoded_sample)
        offsets = np.cumsum([0, *(len(levels) for levels in self.encoder.categories_)])
        categorical_components = np.add.reduceat(differences, offsets[:-1], axis=1).mean(axis=0)
        return pd.Series(
            np.concatenate((numeric_components, categorical_components)),
            index=self.numeric + self.categorical,
        ).sort_values(ascending=False, kind="stable")


def vote_weights(distances, weighting):
    if weighting == "uniform":
        return np.ones_like(distances)
    zero = distances == 0
    # Inverse-distance KNN gives the vote to exact matches when they exist.
    inverse = np.divide(1, distances, out=np.zeros_like(distances), where=~zero)
    return np.where(zero.any(axis=1, keepdims=True), zero, inverse)


def positive_probability(train_labels, distances, indices, k, weighting):
    weights = vote_weights(distances[:, :k], weighting)
    return (weights * train_labels[indices[:, :k]]).sum(axis=1) / weights.sum(axis=1)
