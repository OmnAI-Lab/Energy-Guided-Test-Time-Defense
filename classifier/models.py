import random
import numpy as np
import torch
from robustbench.utils import load_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu: str):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        return device

    return torch.device("cpu")


def load_robustbench_model(
    model_name: str,
    dataset: str,
    model_dir: str,
    device,
):
    print(f"Loading RobustBench model: {model_name}")

    model = load_model(
        model_name=model_name,
        dataset=dataset.lower(),
        threat_model="Linf",
        model_dir=model_dir,
    )

    model.eval()
    model.to(device)

    print("Model loaded successfully.")
    return model