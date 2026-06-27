import argparse
import torch
from tqdm import tqdm

from data import get_dataset, make_subset, get_loader
from models import set_seed, get_device, load_robustbench_model
from defense import EnergytransformWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean accuracy evaluation for transform-wrapped model"
    )

    parser.add_argument("--dataset", type=str, choices=["imagenet"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--indices_path", type=str, default=None)

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--transform_alpha", type=float, default=5.0)
    parser.add_argument("--transform_eps", type=float, default=10.0)
    parser.add_argument("--transform_t", type=float, default=1.0)
    parser.add_argument("--transform_steps", type=int, default=1)

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.gpu)

    dataset = get_dataset(args.dataset, args.data_dir)

    dataset, _ = make_subset(
        dataset=dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        indices_path=args.indices_path,
    )

    loader = get_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    base_model = load_robustbench_model(
        model_name=args.model_name,
        dataset=args.dataset,
        model_dir=args.model_dir,
        device=device,
    )

    defended_model = EnergytransformWrapper(
        model=base_model,
        alpha=args.transform_alpha,
        eps=args.transform_eps,
        t=args.transform_t,
        max_iter=args.transform_steps,
    ).to(device)

    defended_model.eval()

    total = 0
    correct = 0

    for images, labels in tqdm(loader, desc="Clean evaluation"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = defended_model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    acc = 100.0 * correct / max(total, 1)

    print("\n==============================")
    print("Clean accuracy")
    print("==============================")
    print(f"Dataset:       {args.dataset}")
    print(f"Model:         {args.model_name}")
    print(f"Samples:       {total}")
    print(f"Clean accuracy:{acc:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    main()