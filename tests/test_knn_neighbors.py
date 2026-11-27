import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from algorithm.KNN.neighbors import EuclideanNeighbors, positive_probability


class TestEuclideanNeighbors(unittest.TestCase):
    def test_matches_dense_one_hot_with_rare_unknown_and_missing_levels(self):
        train = pd.DataFrame({
            "row_id": np.arange(40)[::-1],
            "y": np.arange(40) % 2,
            "count": [np.nan] + [10] * 34 + [9, 7, 15, 18, 2],
            "kind": ["common"] * 38 + ["rare", "other"],
        })
        query = pd.DataFrame({"count": [10, 10, np.nan, 18],
                              "kind": ["common", None, "unseen", "rare"]})
        search = EuclideanNeighbors(train, ["count"], ["kind"])
        d, idx = search.neighbors(query, batch_size=2)

        def dense(frame):
            numeric = search.scaler.transform(frame[["count"]])
            one_hot = search.encoder.transform(search.categories(frame)).toarray()
            return np.column_stack((numeric, one_hot))

        all_d = cdist(dense(query), dense(search.train))
        expected = np.argsort(all_d, axis=1, kind="stable")[:, :31]
        np.testing.assert_array_equal(idx, expected)
        np.testing.assert_allclose(d, np.take_along_axis(all_d, expected, axis=1))
        # The 35 exact matches share a boundary at k=31. Row ID resolves it.
        np.testing.assert_array_equal(search.train.iloc[idx[0]]["row_id"], np.arange(5, 36))
        np.testing.assert_allclose(d[1], np.sqrt(2))
        np.testing.assert_allclose(d[2], 1)

    def test_exact_matches_receive_all_inverse_distance_weight(self):
        labels = np.array([0, 1, 1, 1, 1])
        idx = np.arange(5)[None, :]
        distances = np.array([[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]], dtype=float)
        idx = np.repeat(idx, repeats=2, axis=0)
        np.testing.assert_allclose(
            positive_probability(labels, distances, idx, 5, "distance"),
            [0.5, 1 - 1 / (1 + 1/2 + 1/3 + 1/4 + 1/5)],
        )
        np.testing.assert_allclose(positive_probability(labels, distances, idx, 5, "uniform"), 0.8)
