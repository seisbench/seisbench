"""
Convert the original SeisT detection/picking weights (https://github.com/senli1073/SeisT, ``pretrained/seist_*_dpk_*.pth``)
into the SeisBench weights format.

The original checkpoints were trained on the DiTing dataset at 50 Hz with windows of 8192 samples.
The state dict layout of :py:class:`seisbench.models.SeisT` matches the original implementation, only the
``backbone.`` prefix needs to be added.

Example:
    python seist_original_conversion.py --variant m --checkpoint seist_m_dpk_diting.pth --output weights/seist/diting_m
"""

import argparse
import json
from pathlib import Path

import torch

import seisbench.models as sbm

DOCSTRING = (
    "Original SeisT-{variant_upper} detection and picking weights from Li et al. (2024, "
    "https://doi.org/10.1109/TGRS.2024.3371503), trained on the DiTing dataset (Zhao et al., 2023) at 50 Hz "
    "with windows of 8192 samples. Originally published under the MIT License at "
    "https://github.com/senli1073/SeisT/blob/main/pretrained/seist_{variant}_dpk_diting.pth .\n"
    "When using this model, please reference Li et al. (2024) and the SeisBench publications listed at "
    "https://github.com/seisbench/seisbench\n\n"
    "Converted to SeisBench by Sen Li (senli.1073@gmail.com)"
)


def convert(
    variant: str,
    checkpoint: Path,
    output: Path,
    version: str = "1",
    seisbench_requirement: str = "0.12.5",
) -> None:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_dict" in state_dict:
        state_dict = state_dict["model_dict"]
    state_dict = {
        "backbone." + k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    model = sbm.SeisT(
        variant=variant, in_samples=8192, sampling_rate=50, norm="std", phases="PS"
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(
        output,
        weights_docstring=DOCSTRING.format(
            variant=variant, variant_upper=variant.upper()
        ),
        version_str=version,
    )

    path_json = output.with_name(f"{output.name}.json.v{version}")
    with open(path_json) as f:
        metadata = json.load(f)
    # Same key layout as the files in the SeisBench weights repository
    metadata = {
        "docstring": metadata["docstring"],
        "model_args": metadata["model_args"],
        "seisbench_requirement": seisbench_requirement,
        "version": version,
        "default_args": {"overlap": 4096, "blinding": [250, 250]},
    }
    with open(path_json, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {path_json} and {output.with_name(f'{output.name}.pt.v{version}')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["s", "m", "l"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path without suffix"
    )
    parser.add_argument("--version", default="1")
    parser.add_argument(
        "--seisbench-requirement",
        default="0.12.5",
        help="Minimal SeisBench version written to the metadata",
    )
    args = parser.parse_args()
    convert(
        args.variant,
        args.checkpoint,
        args.output,
        args.version,
        args.seisbench_requirement,
    )
