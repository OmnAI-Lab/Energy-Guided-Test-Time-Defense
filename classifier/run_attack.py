import os
import argparse

import torch
from tqdm import tqdm

from data import get_dataset, make_subset, get_loader
from models import set_seed, get_device, load_robustbench_model
from defense import EnergytransformWrapper
from apgd_dlr import apgd_t_dlr_restarts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal APGD-T/DLR evaluation for transfer and BPDA-style attacks"
    )

    # Data/model
    parser.add_argument("--dataset", type=str, choices=["imagenet"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    # Attack mode
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["transfer", "bpda"],
        required=True,
        help="transfer: attack base model. bpda: attack transform-wrapped model.",
    )

    # Subset
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument(
        "--indices_path",
        type=str,
        required=True,
        help="Path to save/load fixed subset indices. Use same path for both attacks.",
    )

    # Runtime
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=942)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # APGD-T / DLR
    parser.add_argument("--eps", type=float, default=4.0 / 255.0)
    parser.add_argument("--apgd_steps", type=int, default=100)
    parser.add_argument("--apgd_restarts", type=int, default=5)

    # transform
    parser.add_argument("--transform_alpha", type=float, default=5.0)
    parser.add_argument("--transform_eps", type=float, default=10.0)
    parser.add_argument("--transform_t", type=float, default=1.0)
    parser.add_argument("--transform_steps", type=int, default=1)

    # Output
    parser.add_argument("--save_path", type=str, required=True)

    return parser.parse_args()


@torch.no_grad()
def correct_mask(model, images, labels):
    logits = model(images)
    preds = logits.argmax(dim=1)
    return preds == labels


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.gpu)

    print("\n==============================")
    print("Experiment configuration")
    print("==============================")
    print(f"Dataset:          {args.dataset}")
    print(f"Model:            {args.model_name}")
    print(f"Attack type:      {args.attack_type}")
    print(f"Num samples:      {args.num_samples}")
    print(f"Epsilon:          {args.eps}")
    print(f"APGD steps:       {args.apgd_steps}")
    print(f"APGD restarts:    {args.apgd_restarts}")
    print(f"transform alpha:   {args.transform_alpha}")
    print(f"transform eps L2:  {args.transform_eps}")
    print(f"transform steps:   {args.transform_steps}")
    print("==============================\n")

    dataset = get_dataset(args.dataset, args.data_dir)

    dataset, subset_indices = make_subset(
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

    base_model.eval()
    defended_model.eval()

    if args.attack_type == "transfer":
        attack_model = base_model

        print("Running TRANSFER APGD-T/DLR:")
        print("  Attack model:     base model")
        print("  Evaluation model: transform-wrapped defended model\n")

    elif args.attack_type == "bpda":
        attack_model = defended_model

        print("Running BPDA-style APGD-T/DLR:")
        print("  Attack model:     transform-wrapped defended model")
        print("  Evaluation model: transform-wrapped defended model")
        print("  BPDA behavior comes from detached transform updates.\n")

    else:
        raise ValueError(f"Unknown attack_type: {args.attack_type}")

    all_adv_images = []
    all_labels = []
    all_clean_correct = []
    all_adv_correct = []

    for images, labels in tqdm(loader, desc=f"{args.attack_type} APGD-T/DLR"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_correct = correct_mask(defended_model, images, labels)

        adv_images, best_images = apgd_t_dlr_restarts(
            model=attack_model,
            x=images,
            y=labels,
            norm="Linf",
            eps=args.eps,
            n_iter=args.apgd_steps,
            n_restarts=args.apgd_restarts,
            eot_iter=0,
            early_stop=False,
            verbose=False,
        )

        adv_images = adv_images.detach()

        adv_correct = correct_mask(defended_model, adv_images, labels)

        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())
        all_clean_correct.append(clean_correct.cpu())
        all_adv_correct.append(adv_correct.cpu())

    adv_images = torch.cat(all_adv_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    clean_correct = torch.cat(all_clean_correct, dim=0)
    adv_correct = torch.cat(all_adv_correct, dim=0)

    clean_acc = 100.0 * clean_correct.float().mean().item()
    robust_acc = 100.0 * adv_correct.float().mean().item()

    result = {
        "attack_type": args.attack_type,
        "dataset": args.dataset,
        "model_name": args.model_name,

        "adv_images": adv_images,
        "labels": labels,
        "subset_indices": subset_indices.cpu() if subset_indices is not None else None,

        "clean_correct": clean_correct,
        "adv_correct": adv_correct,

        "clean_acc": clean_acc,
        "robust_acc": robust_acc,

        "eps": args.eps,
        "apgd_steps": args.apgd_steps,
        "apgd_restarts": args.apgd_restarts,

        "transform_alpha": args.transform_alpha,
        "transform_eps": args.transform_eps,
        "transform_t": args.transform_t,
        "transform_steps": args.transform_steps,
    }

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(result, args.save_path)

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Saved results to: {args.save_path}")
    print(f"Clean accuracy on subset:      {clean_acc:.2f}%")
    print(f"Robust accuracy ({args.attack_type}): {robust_acc:.2f}%")
    print("==============================\n")


if __name__ == "__main__":
    main()