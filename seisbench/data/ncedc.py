import os
import random
from pathlib import Path

import h5py

import seisbench
import seisbench.util
from .base import BenchmarkDataset, WaveformDataWriter


class NCEDC(BenchmarkDataset):
    """
    NCEDC waveform archive (2021-2023).

    Splits are set using standard random sampling of :py:class: BenchmarkDataset.
    """

    def __init__(self, **kwargs):
        citation = (
            "NCEDC : Northern California Earthquake Center."
            "NCEDC, J., 2014. Northern California Earthquake Data Center. UC Berkeley Seismological Laboratory. Da-taset."
        )

        seisbench.logger.warning(
            "Check available storage and memory before downloading and general use "
            "of NCEDC dataset. "
            "Dataset size: waveforms.hdf5 ~70Gb"
        )

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        download_instructions = (
            "Please download NCEDC following the instructions at https://huggingface.co/datasets/AI4EPS/CEED. "
            "Provide the locations of the NCEDC unpacked files (*.h5) in the "
            "download_kwargs argument 'basepath'."
            "This step is only necessary the first time NCEDC is loaded."
        )

        if basepath is None:
            raise ValueError(
                "No cached version of NCEDC found. " + download_instructions
            )

        basepath = Path(basepath)

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "sampling_rate": 100,
            "unit": "1e-6m/s",
            "instrument_response": "not restituted",
        }

        h5_files = [os.path.join(root, file) for root, dirs, files in os.walk(basepath) for file in files if
                    file.endswith('.h5')]

        all_metadata = []
        all_data = []

        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                groups = list(f.keys())
                print(len(groups))
                for group in groups:
                    datasets = f[group]
                    begin_time = datasets.attrs.get("begin_time")
                    depth_km = datasets.attrs.get("depth_km")
                    end_time = datasets.attrs.get("end_time")
                    event_id = datasets.attrs.get("event_id")
                    event_time = datasets.attrs.get("event_time")
                    event_time_index = datasets.attrs.get("event_time_index")
                    latitude = datasets.attrs.get("latitude")
                    longitude = datasets.attrs.get("longitude")
                    magnitude = datasets.attrs.get("magnitude")
                    magnitude_type = datasets.attrs.get("magnitude_type")
                    nt = datasets.attrs.get("nt")
                    nx = datasets.attrs.get("nx")
                    sampling_rate = datasets.attrs.get("sampling_rate")
                    source = datasets.attrs.get("source")

                    groups2 = list(datasets.keys())
                    for group2 in groups2:
                        data = datasets[group2][()]
                        data = data[[2, 1, 0]]  # From ENZ to ZNE

                        azimuth = datasets[group2].attrs.get('azimuth', 0.0)
                        back_azimuth = datasets[group2].attrs.get('back_azimuth', 0.0)
                        component = datasets[group2].attrs.get('component', '')
                        distance_km = datasets[group2].attrs.get('distance_km', 0.0)
                        dt_s = datasets[group2].attrs.get('dt_s', 0.0)
                        elevation_m = datasets[group2].attrs.get('elevation_m', 0.0)
                        event_id_list = datasets[group2].attrs.get('event_id', [])
                        instrument = datasets[group2].attrs.get('instrument', '')
                        local_depth_m = datasets[group2].attrs.get('local_depth_m', 0.0)
                        location = datasets[group2].attrs.get('location', '')
                        network = datasets[group2].attrs.get('network', '')
                        phase_index = datasets[group2].attrs.get('phase_index', [])
                        phase_picking_channel = datasets[group2].attrs.get('phase_picking_channel', [])
                        phase_polarity = datasets[group2].attrs.get('phase_polarity', [])
                        phase_remark = datasets[group2].attrs.get('phase_remark', [])
                        phase_score = datasets[group2].attrs.get('phase_score', [])
                        phase_status = datasets[group2].attrs.get('phase_status', '')
                        phase_time = datasets[group2].attrs.get('phase_time', [])
                        phase_type = datasets[group2].attrs.get('phase_type', [])
                        snr = datasets[group2].attrs.get('snr', [])
                        station = datasets[group2].attrs.get('station', '')
                        unit = datasets[group2].attrs.get('unit', '')

                        if len(phase_index) == 2:
                            # # 根据phase_type进行捆绑
                            for i, p_type in enumerate(phase_type):
                                if p_type == 'P':
                                    p_index = phase_index[i]
                                    p_time = phase_time[i]
                                    p_polarity = phase_polarity[i]
                                elif p_type == 'S':
                                    s_index = phase_index[i]
                                    s_time = phase_time[i]

                            metadata = {
                                "trace_name": group + '/' + group2,
                                "trace_category": "earthquake",
                                "trace_p_arrival_sample": p_index,
                                "trace_s_arrival_sample": s_index,
                                "trace_p_arrival": p_time,
                                "trace_s_arrival": s_time,
                                "trace_p_status": phase_status,
                                "trace_snr_db": snr,
                                "trace_instrument": instrument,
                                "trace_polarity": p_polarity,
                                "station_network_code": network,
                                "station_code": station,
                                "source_magnitude": magnitude,
                                "source_magnitude_type": magnitude_type,
                                "source_id": event_id,
                                "source_latitude": latitude,
                                "source_longitude": longitude,
                                "azimuth": azimuth,
                                "sampling_rate": sampling_rate,
                                "begin_time": begin_time,
                                "end_time": end_time,
                            }

                            all_metadata.append(metadata)
                            all_data.append(data)

        # 生成随机索引
        indices = list(range(len(all_metadata)))
        random.shuffle(indices)

        num_samples = len(all_metadata)
        num_train = int(num_samples * 0.7)
        num_dev = int(num_samples * 0.1)
        num_test = num_samples - num_train - num_dev

        # 用于存储打乱后索引对应的划分信息
        split_info = [None] * num_samples

        for i, index in enumerate(indices):
            if i < num_train:
                split_info[index] = 'train'
            elif i < num_train + num_dev:
                split_info[index] = 'dev'
            else:
                split_info[index] = 'test'

        # 恢复顺序并添加划分信息
        for i in range(num_samples):
            all_metadata[i]['split'] = split_info[i]
            writer.add_trace(all_metadata[i], all_data[i])