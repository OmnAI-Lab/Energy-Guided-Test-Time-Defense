from contextlib import suppress
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from autoattack import AutoAttack


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model, normalize_fn, resize_fn, classifier):
        super().__init__()

        if isinstance(clip_model, torch.nn.DataParallel):
            self.clip_model = clip_model.module
        else:
            self.clip_model = clip_model

        self.normalize = normalize_fn
        self.resize = resize_fn if resize_fn is not None else lambda x: x
        self.register_buffer("classifier", classifier)

    def forward(self, data_unnorm):
        data_unnorm = self.resize(data_unnorm)
        data_norm = self.normalize(data_unnorm)

        features = self.clip_model.encode_image(data_norm)
        features = F.normalize(features, dim=-1)

        logits = 100.0 * features @ self.classifier
        return logits


def energy_x(logits, t=1.0):
    return -t * torch.logsumexp(logits / t, dim=1)


def transform_images(images, model_forward_fn, alpha=1.0, eps=10.0, t=1.0, max_iters=10):
    eps = eps / 255.0

    with torch.enable_grad():
        original = images.detach()
        transformed = images.clone().detach()
        transformed.requires_grad = True

        for _ in range(max_iters):
            logits = model_forward_fn(transformed)
            energy_vec = energy_x(logits, t=t)

            grad = torch.autograd.grad(
                energy_vec.sum(),
                transformed,
                create_graph=False,
                retain_graph=False,
            )[0]

            grad_norm = torch.norm(
                grad,
                p=2,
                dim=(1, 2, 3),
                keepdim=True,
            ).clamp(min=1e-8)

            with torch.no_grad():
                transformed = transformed - alpha * grad / grad_norm
                transformed = torch.clamp(transformed, 0.0, 1.0)

                perturbation = transformed - original
                perturbation = torch.clamp(perturbation, -eps, eps)

                transformed = torch.clamp(original + perturbation, 0.0, 1.0)
                transformed = transformed.detach()

            transformed.requires_grad = True

    return transformed.detach()


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    autocast = torch.cuda.amp.autocast if amp else suppress

    with torch.no_grad(), autocast():
        zeroshot_weights = []

        for classname in tqdm(classnames, desc="Building zero-shot classifier"):
            if type(templates) == dict:
                texts = templates[classname]
            elif type(templates) == list:
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")

            texts = tokenizer(texts).to(device)

            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)

    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n
        for k in topk
    ]


def run_classification(
    model,
    classifier,
    dataloader,
    device,
    normalize=None,
    resize=None,
    amp=True,
    attack_config=None,
    transform_config=None,
    classifier_transform=None,
):
    assert normalize is not None

    if attack_config is None:
        attack_config = {
            "attack": "none",
            "bs": 64,
            "n_samples": -1,
        }

    max_samples = attack_config["n_samples"]
    autocast = torch.cuda.amp.autocast if amp else suppress

    use_transform = False
    if transform_config is not None and transform_config.get("enabled", False):
        use_transform = True
        transform_alpha = transform_config.get("alpha", 1.0)
        transform_eps = transform_config.get("eps", 10.0)
        transform_t = transform_config.get("t", 1.0)
        transform_max_iters = transform_config.get("max_iters", 5)

        print(
            f"[Transform enabled] alpha={transform_alpha}, "
            f"eps={transform_eps}, t={transform_t}, max_iters={transform_max_iters}"
        )

    if classifier_transform is None:
        classifier_transform = classifier

    clean_pred = []
    clean_true = []
    aux_pred = []

    adv_pred = []
    adv_true = []

    encoder = ImageEncoderWrapper(model, normalize, resize, classifier)
    encoder = encoder.to(device)
    encoder.eval()

    encoder_transform = ImageEncoderWrapper(model, normalize, resize, classifier_transform)
    encoder_transform = encoder_transform.to(device)
    encoder_transform.eval()

    def _forward_unnorm(data_unnorm):
        return encoder(data_unnorm)

    def _forward_unnorm_transform(data_unnorm):
        return encoder_transform(data_unnorm)

    attack_str = attack_config["attack"]
    should_attack = attack_str != "none"

    if should_attack:
        bs = attack_config["bs"]
        norm = attack_config["norm"]
        eps = attack_config["eps"] / 255.0
        save_adv = attack_config.get("save_adv", False)

        if attack_str.lower() != "aa":
            raise ValueError(f"Unsupported attack: {attack_str}")

        n_classes = classifier.shape[1]
        attacks_to_run = ["apgd-ce", "apgd-t"] if n_classes > 2 else ["apgd-ce"]

        attack = AutoAttack(
            _forward_unnorm,
            norm=norm,
            eps=eps,
            attacks_to_run=attacks_to_run,
            version="custom",
            verbose=True,
            device=device,
        )

        attack.apgd.n_iter = attack_config.get("iterations", 100)
        attack.apgd_targeted.n_iter = attack_config.get("iterations", 100)

        all_images = []
        all_targets = []

        collected = 0

        for batch in dataloader:
            images = batch[0]
            targets = batch[1]

            if max_samples > 0 and collected >= max_samples:
                break

            if max_samples > 0:
                remaining = max_samples - collected
                images = images[:remaining]
                targets = targets[:remaining]

            all_images.append(images)
            all_targets.append(targets)

            collected += images.shape[0]

        all_images = torch.cat(all_images, dim=0).to(device)
        all_targets = torch.cat(all_targets, dim=0).to(device)

        assert 0.0 <= all_images.min() and all_images.max() <= 1.0, (
            all_images.min(),
            all_images.max(),
        )

        print(f"[n samples] {len(all_images)}")
        print("Starting AutoAttack...")

        if save_adv:
            path = attack_config.get("save_adv_path")
            if path is None:
                raise ValueError("save_adv=True but save_adv_path is None")

            path_obj = Path(path)

            if path_obj.exists():
                print(f"Loading adversarial examples from {path}")
                data = torch.load(path, map_location="cpu")
                images_adv = data["images"].to(device)
                labels_adv = data["labels"].to(device)
            else:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                images_adv = attack.run_standard_evaluation(all_images, all_targets, bs=bs)
                labels_adv = all_targets

                torch.save(
                    {
                        "images": images_adv.detach().cpu(),
                        "labels": labels_adv.detach().cpu(),
                    },
                    path,
                )
                print(f"Saved adversarial examples to {path}")
        else:
            images_adv = attack.run_standard_evaluation(all_images, all_targets, bs=bs)
            labels_adv = all_targets

        print("Getting adversarial logits...")

        for i in range(0, len(images_adv), bs):
            batch = images_adv[i:i + bs].to(device)
            batch_targets = labels_adv[i:i + bs].to(device)

            if use_transform:
                with torch.enable_grad():
                    batch = transform_images(
                        batch,
                        _forward_unnorm_transform,
                        alpha=transform_alpha,
                        eps=transform_eps,
                        t=transform_t,
                        max_iters=transform_max_iters,
                    )

            with torch.no_grad():
                logits = _forward_unnorm(batch)

            adv_pred.append(logits.float().cpu())
            adv_true.append(batch_targets.cpu())

        adv_pred = torch.cat(adv_pred)
        adv_true = torch.cat(adv_true)

    else:
        print("Running clean evaluation only")
        adv_pred, adv_true = None, None

    with torch.no_grad():
        n = 0

        for images, target in tqdm(dataloader, desc="Clean evaluation"):
            if max_samples > 0 and n >= max_samples:
                break

            if max_samples > 0:
                remaining = max_samples - n
                images = images[:remaining]
                target = target[:remaining]

            images = images.to(device)
            target = target.to(device)

            n += images.shape[0]

            if use_transform:
                with torch.enable_grad():
                    images = transform_images(
                        images,
                        _forward_unnorm_transform,
                        alpha=transform_alpha,
                        eps=transform_eps,
                        t=transform_t,
                        max_iters=transform_max_iters,
                    )

            with autocast():
                logits = _forward_unnorm(images)
                aux_logits = _forward_unnorm_transform(images)

            clean_true.append(target.cpu())
            clean_pred.append(logits.float().cpu())
            aux_pred.append(aux_logits.float().cpu())

    clean_pred = torch.cat(clean_pred)
    clean_true = torch.cat(clean_true)
    aux_pred = torch.cat(aux_pred)

    if max_samples > 0:
        clean_pred = clean_pred[:max_samples]
        clean_true = clean_true[:max_samples]

    print(f"[n samples] {len(clean_pred)}")

    return clean_pred, clean_true, adv_pred, adv_true


def evaluate(
    model,
    dataloader,
    tokenizer,
    classnames,
    templates,
    device,
    normalize=None,
    resize=None,
    amp=True,
    verbose=False,
    save_clf=None,
    load_clfs=[],
    attack_config=None,
    transform_config=None,
    transform_classnames=None,
    transform_templates=None,
):
    assert normalize is not None

    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location="cpu") / n

        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location="cpu") / n

        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(
            model,
            tokenizer,
            classnames,
            templates,
            device,
            amp=amp,
        )

        if save_clf is not None:
            save_path = Path(save_clf)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(classifier.detach().cpu(), save_path)
            print(f"Saved classifier to {save_path}")

    if transform_config is not None and transform_config.get("enabled", False):
        if transform_classnames is not None:
            if transform_templates is None:
                transform_templates = templates

            print(
                f"[Transform classifier] Building separate transform classifier "
                f"with {len(transform_classnames)} classes"
            )

            classifier_transform = zero_shot_classifier(
                model,
                tokenizer,
                transform_classnames,
                transform_templates,
                device,
                amp=amp,
            )
        else:
            print("[Transform classifier] Using dataset classifier")
            classifier_transform = classifier
    else:
        classifier_transform = classifier
    
    
    logits, target, adv_logits, adv_target = run_classification(
        model,
        classifier,
        dataloader,
        device,
        normalize=normalize,
        resize=resize,
        amp=amp,
        attack_config=attack_config,
        transform_config=transform_config,
        classifier_transform=classifier_transform,
    )

    pred = logits.argmax(axis=1)

    clean_acc, = accuracy(logits, target, topk=(1,))

    if adv_logits is None:
        adv_acc = float("nan")
    else:
        adv_acc, = accuracy(adv_logits, adv_target, topk=(1,))

    mean_per_class_recall = balanced_accuracy_score(target, pred)

    print(f"[clean_accuracy] {clean_acc * 100:.2f}")
    print(f"[adv_accuracy] {adv_acc * 100:.2f}")

    return {
        "clean_accuracy": clean_acc,
        "adv_accuracy": adv_acc,
        "mean_per_class_recall": mean_per_class_recall,
    }