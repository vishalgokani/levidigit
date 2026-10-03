"""nnU-Net predictor construction and execution."""

from __future__ import annotations

from pathlib import Path

import torch

from .constants import MODEL_CHECKPOINT
from .model import validate_model_folder


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        print(
            f"Inference device: cuda ({torch.cuda.get_device_name(index)}; "
            f"PyTorch {torch.__version__}; CUDA {torch.version.cuda})",
            flush=True,
        )
        return torch.device("cuda")
    print("Inference device: cpu (CUDA is not available).", flush=True)
    return torch.device("cpu")


def run_prediction(model_folder: Path, input_dir: Path, output_dir: Path) -> list[int]:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    folds = validate_model_folder(model_folder)
    device = resolve_device()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=tuple(folds),
        checkpoint_name=MODEL_CHECKPOINT,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return folds

