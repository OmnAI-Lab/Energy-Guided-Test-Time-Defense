import argparse
import json
import os
import random

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize

from clip_benchmark.datasets.builder import (
    build_dataset,
    get_dataset_collate_fn,
    get_dataset_default_task,
)

from clip_benchmark.models import load_clip, MODEL_TYPES
from clip_benchmark.metrics import zeroshot_classification


def parse_args():
    parser = argparse.ArgumentParser("Minimal CLIP zero-shot + AA evaluation")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=100)

    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--model_type", type=str, default="open_clip", choices=MODEL_TYPES)
    parser.add_argument("--model_cache_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_false", dest="amp", default=False)

    parser.add_argument("--attack", type=str, default="none", choices=["none", "aa"])
    parser.add_argument("--norm", type=str, default="Linf")
    parser.add_argument("--eps", type=float, default=4.0, help="Pixel epsilon, e.g. 4 means 4/255")
    parser.add_argument("--iterations_adv", type=int, default=100)

    parser.add_argument("--save_adv", action="store_true")
    parser.add_argument("--save_adv_path", type=str, default="./clip_results/adv_examples.pt")

    parser.add_argument("--purify", action="store_true")
    parser.add_argument("--purify_alpha", type=float, default=2.5)
    parser.add_argument("--purify_eps", type=float, default=10.0)
    parser.add_argument("--purify_t", type=float, default=1.0)
    parser.add_argument("--purify_max_iters", type=int, default=5)

    parser.add_argument("--output", type=str, default="./clip_results/result.json")

    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_adv_path), exist_ok=True)

    print("\n==============================")
    print("CLIP zero-shot evaluation")
    print("==============================")
    print(f"Dataset:       {args.dataset}")
    print(f"Dataset root:  {args.dataset_root}")
    print(f"Model:         {args.model}")
    print(f"Pretrained:    {args.pretrained}")
    print(f"Attack:        {args.attack}")
    print(f"n_samples:     {args.n_samples}")
    print(f"eps:           {args.eps}/255")
    print(f"Purify:        {args.purify}")
    print("==============================\n")

    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        device=device,
    )

    model.eval()

    # Keep images unnormalized for attacks. Normalize inside wrapper.
    if ("cifar10" in args.dataset) or ("cifar100" in args.dataset) or ("stl10" in args.dataset):
        transform_unnorm = transforms.ToTensor()
        resize = Resize(
            size=224,
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=None,
        )
    else:
        transform_unnorm = Compose(transform.transforms[:-1])
        resize = None

    normalize = transform.transforms[-1]
    del transform

    task = get_dataset_default_task(args.dataset)
    assert task == "zeroshot_classification", f"Expected zeroshot_classification, got {task}"

    dataset = build_dataset(
        dataset_name=args.dataset,
        root=args.dataset_root,
        transform=transform_unnorm,
        split=args.split,
        download=True,
        language=["en"],
        task=task,
    )

    if args.n_samples > 0:
        dataset = dataset.shuffle(10000, initial=10000, rng=random.Random(args.seed))

    collate_fn = get_dataset_collate_fn(args.dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    classnames = dataset.classes
    templates = dataset.templates

    attack_config = {
        "attack": args.attack,
        "norm": args.norm,
        "eps": args.eps,
        "iterations": args.iterations_adv,
        "bs": args.batch_size,
        "n_samples": args.n_samples,
        "save_adv": args.save_adv,
        "save_adv_path": args.save_adv_path if args.save_adv else None,
    }

    purify_config = None
    if args.purify:
        purify_config = {
            "enabled": True,
            "alpha": args.purify_alpha,
            "eps": args.purify_eps,
            "t": args.purify_t,
            "max_iters": args.purify_max_iters,
        }

    metrics = zeroshot_classification.evaluate(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        normalize=normalize,
        resize=resize,
        device=device,
        amp=args.amp,
        verbose=True,
        attack_config=attack_config,
        purify_config=purify_config,
        labels_21k=None,
        combine21k=False,
    )

    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "attack": args.attack,
        "eps": args.eps,
        "iterations_adv": args.iterations_adv,
        "n_samples": args.n_samples,
        "purify": args.purify,
        "purify_alpha": args.purify_alpha if args.purify else None,
        "purify_eps": args.purify_eps if args.purify else None,
        "purify_steps": args.purify_max_iters if args.purify else None,
        "metrics": metrics,
    }

    with open(args.output, "w") as f:
        json.dump(dump, f, indent=2)

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(metrics)
    print(f"Saved to: {args.output}")
    print("==============================\n")


if __name__ == "__main__":
    main()