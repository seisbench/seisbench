"""
This is a minimalist benchmark to compare different versions of the annotate function
"""

import time

import numpy as np
import obspy

import seisbench.models as sbm


def main(n_stations: int = 2, n_samples: int = 8640000):
    models = [sbm.PhaseNetLight(), sbm.PhaseNet(), sbm.EQTransformer()]
    # models = [sbm.PhaseNet()]

    data = obspy.Stream()
    for sid in range(n_stations):
        for c in "ZNE":
            data.append(
                obspy.Trace(
                    np.random.rand(n_samples),
                    header={
                        "network": "CX",
                        "station": f"S{sid:04d}",
                        "location": "",
                        "channel": f"HH{c}",
                        "sampling_rate": 100.0,
                    },
                )
            )

    for model in models:
        # model.cuda()
        # Warmup
        models[0].annotate(data[:3])
        t0 = time.time()
        model.annotate(data, batch_size=1024, copy=False)
        t1 = time.time()
        print(model.__class__.__name__, t1 - t0)


if __name__ == "__main__":
    main()
