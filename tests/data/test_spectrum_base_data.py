import pandas as pd
import numpy as np

import seisbench.data as sbd


def test_spectrum_dataset():
    data = sbd.SpectrumDataset(
        "tests/examples/esm_spectrum_example/",
        component_order=(
            "acc_mp_u",
            "acc_mp_v",
            "acc_mp_w",
            "dis_mp_u",
            "dis_mp_v",
            "dis_mp_w",
        ),
    )
    assert isinstance(data.metadata, pd.DataFrame)
    assert isinstance(data.frequencies, np.ndarray)

    assert isinstance(data.get_spectra(0), np.ndarray)
    wc, meta = data.get_sample(0)
    assert isinstance(wc, np.ndarray)
    assert isinstance(meta, dict)
