import numpy as np
import xdas
from unittest.mock import patch

import seisbench.models as sbm


def test_deepsubdas():
    # Simple test to check that the code does not crash
    data = np.random.rand(4000, 2000)

    nt, nd = data.shape
    dt = np.timedelta64(10, "ms")
    dx = 10
    t0 = np.datetime64("2026-01-01 00:00:00").astype("datetime64[ms]")
    t = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    d = {"tie_indices": [0, nd - 1], "tie_values": [0.0, (nd - 1) * dx]}

    da = xdas.DataArray(data, {"time": t, "distance": d})

    model = sbm.DeepSubDAS()

    # Mock function to avoid costly forward call
    def fake_forward(_, x, *args, **kwargs):
        return {"full": x, "P": x, "S": x}

    with patch("seisbench.models.DeepSubDAS.forward", fake_forward):
        callback = sbm.InMemoryCollectionCallback()
        model.annotate(da, callback)
        assert "P" in callback.get_results_dict()
        assert "S" in callback.get_results_dict()
        model.classify(da)
