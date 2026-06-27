
import argparse
import csv
import json
import os
import random
import sys
from copy import copy

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.transforms import Compose, Resize

from clip_benchmark.datasets.builder import (
    build_dataset,
    get_dataset_collate_fn,
    get_dataset_default_task,
    dataset_collection,
    get_dataset_collection_from_file,
)
from clip_benchmark.metrics import zeroshot_classification
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES


def format_latex_row(values, label=""):
    valid_values = [v for v in values if v is not None]

    if not valid_values:
        return f"{label} & " + " & ".join(["--"] * 15)

    avg = sum(valid_values) / len(valid_values)
    formatted = [f"{v * 100:.2f}" if v is not None else "--" for v in values]
    formatted.append(f"{avg * 100:.2f}")

    row = " & ".join(formatted)
    if label:
        row = f"{label} & {row}"

    return row


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser("eval", help="Evaluate")

    parser_eval.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        nargs="+",
        help="Dataset(s), collection name, or path to dataset list file.",
    )
    parser_eval.add_argument(
        "--dataset_root",
        default="root",
        type=str,
        help="Dataset root. Can be template, e.g. datasets/{dataset}.",
    )
    parser_eval.add_argument("--split", type=str, default="test")
    parser_eval.add_argument("--model", type=str, default="ViT-B-32")
    parser_eval.add_argument("--pretrained", type=str, default="openai")
    parser_eval.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        nargs="+",
        help="Model collection, file, or model,pretrained pair.",
    )
    parser_eval.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["zeroshot_classification", "auto"],
    )
    parser_eval.add_argument(
        "--no_amp",
        action="store_false",
        dest="amp",
        default=False,
    )
    parser_eval.add_argument("--num_workers", default=4, type=int)
    parser_eval.add_argument("--seed", default=0, type=int)
    parser_eval.add_argument("--batch_size", default=64, type=int)
    parser_eval.add_argument("--model_cache_dir", default=None, type=str)
    parser_eval.add_argument("--annotation_file", default="", type=str)
    parser_eval.add_argument("--custom_classname_file", default=None, type=str)
    parser_eval.add_argument("--custom_template_file", default=None, type=str)
    parser_eval.add_argument("--language", default="en", type=str, nargs="+")
    parser_eval.add_argument(
        "--output",
        default="result.json",
        type=str,
        help="Output JSON. Can use format fields.",
    )
    parser_eval.add_argument("--quiet", dest="verbose", action="store_false")
    parser_eval.add_argument("--save_clf", default=None, type=str)
    parser_eval.add_argument("--load_clfs", nargs="+", default=[], type=str)
    parser_eval.add_argument("--skip_existing", default=False, action="store_true")
    parser_eval.add_argument("--model_type", default="open_clip", type=str, choices=MODEL_TYPES)
    parser_eval.add_argument("--wds_cache_dir", default=None, type=str)
    parser_eval.add_argument("--n_samples", default=-1, type=int)

    parser_eval.add_argument("--attack", default="none", type=str, choices=["none", "aa"])
    parser_eval.add_argument("--norm", default="Linf", type=str)
    parser_eval.add_argument("--eps", default=4.0, type=float)
    parser_eval.add_argument("--iterations_adv", default=100, type=int)

    parser_eval.add_argument("--save_adv", default=False, action="store_true")
    parser_eval.add_argument("--save_adv_path", default="adv_examples.pt", type=str)

    parser_eval.add_argument("--gpu", type=str, default=None)

    parser_eval.add_argument("--transform", default=False, action="store_true")
    parser_eval.add_argument("--transform_alpha", default=2.5, type=float)
    parser_eval.add_argument("--transform_eps", default=10.0, type=float)
    parser_eval.add_argument("--transform_t", default=1.0, type=float)
    parser_eval.add_argument("--transform_max_iters", default=5, type=int)
    
    
    parser_eval.add_argument(
        "--transform_classifier",
        default="dataset",
        choices=["dataset", "imagenet21k"],
        help="Text classifier used only for the test-time transform."
    )

    parser_eval.add_argument(
        "--imagenet21k_labels",
        default=None,
        type=str,
        help="Path to ImageNet-21k label text file, one classname per line."
    )


    parser_eval.set_defaults(which="eval")

    parser_build = subparsers.add_parser("build", help="Build CSV from evaluations")
    parser_build.add_argument("files", type=str, nargs="+")
    parser_build.add_argument("--output", type=str, default="benchmark.csv")
    parser_build.set_defaults(which="build")

    return parser.parse_args()


def _as_list(x):
    if not x:
        return []
    return [x] if type(x) != list else x


def main():
    args = get_parser_args()

    if args.which == "eval":
        main_eval(args)
    elif args.which == "build":
        build_csv(args)
    else:
        raise ValueError(args.which)


def main_eval(base):
    pretrained_model = _as_list(base.pretrained_model)

    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                models.extend(model_collection[name])
            else:
                model, pretrained = name.split(",")
                models.append((model, pretrained))
    else:
        models = [(base.model, base.pretrained)]

    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            datasets.extend(dataset_collection[name])
        else:
            datasets.append(name)

    if base.verbose:
        print(f"[Models] {models}")
        print(f"[Datasets] {datasets}")

    clean_acc = []
    robust_acc = []

    for model, pretrained in models:
        for i, dataset in enumerate(datasets):
            print(f"\n{i + 1} / {len(datasets)}")

            args = copy(base)
            args.model = model
            args.pretrained = pretrained
            args.dataset = dataset

            c_acc, r_acc = run(args)
            clean_acc.append(c_acc)
            robust_acc.append(r_acc)

    print("\n================ Summary ================")
    print("Clean accuracy:", clean_acc)
    print("Robust accuracy:", robust_acc)

    print("\n================ LaTeX Format ================")
    print(format_latex_row(clean_acc, "Clean"))
    print(format_latex_row(robust_acc, "Robust"))


def run(args):
    print("[args]", args, "\n")

    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"
        if args.gpu is not None:
            torch.cuda.set_device(int(args.gpu))
    else:
        args.device = "cpu"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task

    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    elif args.dataset.startswith("#"):
        print(f"Skip commented dataset {args.dataset}")
        return None, None
    else:
        dataset_name = args.dataset

    if task == "auto":
        task = get_dataset_default_task(dataset_name)

    if task != "zeroshot_classification":
        raise ValueError(f"Unsupported task: {task}")

    pretrained_slug = (
        args.pretrained.split("/")[-1] if os.path.isfile(args.pretrained) else args.pretrained
    )
    pretrained_slug_full_path = (
        args.pretrained.replace("/", "_") if os.path.isfile(args.pretrained) else args.pretrained
    )
    dataset_slug = dataset_name.replace("/", "_")
    model_slug = args.model.replace("/", "_")

    output = args.output.format(
        model=model_slug,
        attack=args.attack,
        eps=str(int(args.eps)),
        iterations=args.iterations_adv,
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        task=task,
        dataset=dataset_slug,
        n_samples=args.n_samples,
        language=args.language,
        bs=args.batch_size,
    )

    if os.path.exists(output) and args.skip_existing:
        print(f"Skip {output}, exists already.")
        return None, None

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.verbose:
        print(f"[Dataset] {args.dataset}")
        print(f"[Task] {task}")
        print(f"[Model] {args.model}")
        print(f"[Pretrained] {args.pretrained}")
        print(f"[Output] {output}")

    dataset_root = args.dataset_root.format(
        dataset=dataset_name,
        dataset_cleaned=dataset_name.replace("/", "-"),
    )

    model, transform, tokenizer = load_clip(
        model_type=args.model_type,
        model_name=args.model,
        pretrained=args.pretrained,
        cache_dir=args.model_cache_dir,
        device=args.device,
    )
    model.eval()

    if ("cifar10" in args.dataset) or ("cifar100" in args.dataset) or ("stl10" in args.dataset):
        transform_unnorm = transforms.transforms.ToTensor()
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

    print(f"[Transform unnorm] {transform_unnorm}")
    print(f"[Normalize] {normalize}")

    dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform_unnorm,
        split=args.split,
        annotation_file=args.annotation_file,
        download=True,
        language=args.language,
        task=task,
        custom_template_file=args.custom_template_file,
        custom_classname_file=args.custom_classname_file,
        wds_cache_dir=args.wds_cache_dir,
    )

    if args.n_samples > 0 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(10000, initial=10000, rng=random.Random(args.seed))

    collate_fn = get_dataset_collate_fn(args.dataset)

    if args.verbose:
        try:
            print(f"Dataset size: {len(dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        print(f"Dataset split: {args.split}")
        if hasattr(dataset, "classes") and dataset.classes:
            print(f"Dataset classes: {dataset.classes[:20]}...")
            print(f"Dataset number of classes: {len(dataset.classes)}")

    if args.dataset.startswith("wds/"):
        dataloader = torch.utils.data.DataLoader(
            dataset.batched(args.batch_size),
            batch_size=None,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
    classnames = dataset.classes if hasattr(dataset, "classes") else None

    assert zeroshot_templates is not None and classnames is not None, (
        "Dataset does not support zero-shot classification"
    )

    attack_config = {
        "attack": args.attack,
        "norm": args.norm,
        "eps": args.eps,
        "iterations": args.iterations_adv,
        "bs": args.batch_size,
        "n_samples": args.n_samples,
        "save_adv": args.save_adv,
        "save_adv_path": args.save_adv_path.format(
            model=model_slug,
            pretrained=pretrained_slug,
            dataset=dataset_slug,
            n_samples=args.n_samples,
            attack=args.attack,
            eps=str(int(args.eps)),
            iterations=args.iterations_adv,
        ) if args.save_adv else None,
    }

    transform_config = {
        "enabled": args.transform,
        "alpha": args.transform_alpha,
        "eps": args.transform_eps,
        "t": args.transform_t,
        "max_iters": args.transform_max_iters,
    } if args.transform else None

    print(f"Attack config: {attack_config}")
    print(f"Transform config: {transform_config}")

    clf_saved = args.save_clf.format(model=model_slug, dataset=dataset_slug) if args.save_clf else None

    
    transform_classnames = None
    transform_templates = None

    if args.transform and args.transform_classifier == "imagenet21k":
        if args.imagenet21k_labels is None:
            raise ValueError(
                "--transform_classifier imagenet21k requires --imagenet21k_labels"
            )

        with open(args.imagenet21k_labels, "r") as f:
            transform_classnames = [line.strip() for line in f if line.strip()]

        transform_templates = zeroshot_templates

        print(f"[Transform classifier] ImageNet-21k labels: {len(transform_classnames)} classes")

    elif args.transform and args.transform_classifier == "dataset":
        print("[Transform classifier] Using dataset classnames")

    metrics = zeroshot_classification.evaluate(
        model,
        dataloader,
        tokenizer,
        classnames,
        zeroshot_templates,
        normalize=normalize,
        resize=resize,
        device=args.device,
        amp=args.amp,
        verbose=args.verbose,
        save_clf=clf_saved,
        load_clfs=args.load_clfs,
        attack_config=attack_config,
        transform_config=transform_config,
        transform_classnames=transform_classnames,
        transform_templates=transform_templates,
    )

    
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
        "attack": args.attack,
        "iterations_adv": args.iterations_adv,
        "eps": args.eps,
        "norm": args.norm,
        "transform": args.transform,
        "transform_alpha": args.transform_alpha if args.transform else None,
        "transform_eps": args.transform_eps if args.transform else None,
        "transform_max_iters": args.transform_max_iters if args.transform else None,
        "transform_classifier": args.transform_classifier if args.transform else None,
        "imagenet21k_labels": args.imagenet21k_labels if args.transform_classifier == "imagenet21k" else None,
}

    print(f"Dump results to: {output}")
    with open(output, "w") as f:
        json.dump(dump, f, indent=2)

    return metrics.get("clean_accuracy", None), metrics.get("adv_accuracy", None)


def build_csv(args):
    rows = []

    for file in args.files:
        with open(file, "r") as f:
            rows.append(json.load(f))

    with open(args.output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    sys.exit(main())