"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
from distutils.command import clean
import logging
from contextlib import suppress
from os import name
from pathlib import Path


from numpy import save
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report, balanced_accuracy_score
from autoattack import AutoAttack


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model, normalize_fn, resize_fn, classifier):
        super().__init__()
        # If model is already DataParallel, unwrap it
        if isinstance(clip_model, torch.nn.DataParallel):
            self.clip_model = clip_model.module
        else:
            self.clip_model = clip_model
        self.normalize = normalize_fn
        self.resize = resize_fn if resize_fn is not None else lambda x: x
        # Register classifier as a buffer so it gets copied to each GPU
        self.register_buffer('classifier', classifier)
        
    def forward(self, data_unnorm):
        """Forward pass that DataParallel can distribute"""
        data_unnorm = self.resize(data_unnorm)
        data_norm = self.normalize(data_unnorm)
        features = self.clip_model.encode_image(data_norm)
        features = F.normalize(features, dim=-1)
        # Now classifier is automatically on the same device as features!
        logits = 100. * features @ self.classifier
        return logits
        
def energy_x(logits, t=1.0):
    """Energy function for purification"""
    return -t * torch.logsumexp(logits / t, dim=1)


def purify_images(images, model_forward_fn, alpha=1, eps=10, t=1.0, max_iters=10):
    """
    Purify images using energy-based refinement.
    
    Parameters
    ----------
    images: torch.Tensor
        Input images to purify (B, C, H, W)
    model_forward_fn: callable
        Function that takes images and returns logits
    alpha: float
        Step size for purification
    eps: float
        Maximum L2 perturbation budget
    t: float
        Temperature parameter for energy function
    max_iters: int
        Maximum purification iterations
    
    Returns
    -------
    torch.Tensor
        Purified images with same shape as input
    """
    with torch.enable_grad():
        purified_x = images.clone().detach()#.float()
        purified_x.requires_grad = True
        perturbation_norm = torch.zeros(images.size(0), 1, 1, 1, device=images.device)
        for i in range(max_iters):
            # if i > 0:
            #     alpha = alpha / (i + 1)  # decay step size
            # Early exit if all samples reached eps
            if (perturbation_norm >= eps).all():
                break
            
            logits = model_forward_fn(purified_x)
            energy_vec = energy_x(logits, t=t)
            grad = torch.autograd.grad(energy_vec.sum(), purified_x, create_graph=False)[0]

            # Compute per-sample grad L2 and keep dims for broadcasting
            grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)

            # Mask out samples that already reached eps
            # active = (perturbation_norm < eps).float()
            
            with torch.no_grad():
                purified_x = purified_x - alpha * (grad / grad_norm) #* active
                purified_x = torch.clamp(purified_x, 0, 1)

                perturbation = purified_x - images
                # perturbation = torch.clamp(perturbation, -16/255, 16/255)
                perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), 
                                              p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)

                factor = torch.clamp(eps / (perturbation_norm + 1e-8), max=1.0)
                perturbation = factor * perturbation

                purified_x = torch.clamp(images + perturbation, 0, 1)
                purified_x = purified_x.detach()
            
            purified_x.requires_grad = True

    return purified_x.detach()


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, normalize=None, resize=None, amp=True,
                       attack_config=None, purify_config=None):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    assert normalize is not None
    autocast = torch.cuda.amp.autocast if amp else suppress
    clean_pred = []
    clean_true = []
    pred = []
    true = []
    max_samples = attack_config['n_samples']

    use_purify = False
    if purify_config is not None and purify_config.get('enabled', False):
        use_purify = True
        purify_alpha = purify_config.get('alpha', 1)
        purify_eps = purify_config.get('eps', 5)
        purify_t = purify_config.get('t', 1.0)
        purify_max_iters = purify_config.get('max_iters', 10)
        print(f"[Purification enabled] alpha={purify_alpha}, eps={purify_eps}, t={purify_t}, max_iters={purify_max_iters}")
    
    encoder = ImageEncoderWrapper(model, normalize, resize, classifier)
    if torch.cuda.device_count() > 1:
        print(f"Wrapping encoder with DataParallel for {torch.cuda.device_count()} GPUs")
        encoder = torch.nn.DataParallel(encoder)
    
    encoder = encoder.cuda()
    encoder.eval()
    
    def _forward_unnorm(data_unnorm):
        return encoder(data_unnorm)

    # def _forward_unnorm(data_unnorm):
    #     if resize is not None:
    #         data_unnorm = resize(data_unnorm)
    #     data_norm = normalize(data_unnorm)
    #     features = model.module.encode_image(data_norm)
    #     features = F.normalize(features, dim=-1)

    #     logits = 100. * features @ classifier
    #     return logits

    attack_str = attack_config['attack']
    adv = attack_str != 'none'
    if adv:
        bs = attack_config['bs']
        norm = attack_config['norm']
        eps = attack_config['eps'] / 255.
        save_adv = attack_config.get('save_adv')
        # iterations = attack_config['iterations']
        if attack_str.lower() == 'aa':
            attacks_to_run = (['apgd-ce', 'apgd-t'] if len(dataloader.dataset.classes) > 2 else ['apgd-ce'])
            attack = AutoAttack(
                _forward_unnorm, norm=norm, eps=eps,
                attacks_to_run=attacks_to_run,
                version='custom',
                verbose=True,
                device=device
            )
            all_images, all_targets = [], []
            for i, batch in enumerate(dataloader):
                all_images.append(batch[0])
                all_targets.append(batch[1])
                if (max_samples > 0) and (i >= max_samples // bs + 2):
                    break
            all_images = torch.cat(all_images, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            if max_samples > 0:
                all_images = all_images[:max_samples]
                all_targets = all_targets[:max_samples]
            assert 0. <= all_images.min() and all_images.max() <= 1., f'{all_images.min()} {all_images.max()}'

            print(f'[n samples] {len(all_images)}')
            print(f'starting autoattack..')
            if save_adv:
                print(" [Saving/loading adversarial examples]")
                path = attack_config.get('save_adv_path')
                if Path(path).exists():
                    print(f'Adversarial examples already exist at {path}, skipping autoattack and loading existing adv examples.')
                    data = torch.load(path)
                    images, labels = data['images'], data['labels']
                else:
                    images, labels = attack.run_standard_evaluation(all_images, all_targets, bs=bs, return_labels=True)
                    torch.save({
                        'images': images.detach().cpu(),
                        'labels': labels.detach().cpu(),
                    }, path)
                    print(f'Saved adversarial examples to {path}')
            else:
                images = attack.run_standard_evaluation(all_images, all_targets, bs=bs)

            print('getting logits..')
            with torch.no_grad():
                for i in range(0, len(images), bs):
                    batch = images[i:i + bs].to(device)
                    batch_targets= labels[i:i + bs].to(device)
                    if use_purify:
                        batch = purify_images(
                            batch,
                            _forward_unnorm,
                            alpha=purify_alpha,
                            eps=purify_eps,
                            t=purify_t,
                            max_iters=purify_max_iters
                        )
                    logits = _forward_unnorm(batch)
                    pred.append(logits.float().cpu())
                    true.append(batch_targets.cpu())
                pred = torch.cat(pred)
                true = torch.cat(true)
    else:
        print("Running clean evaluation only")
        pred, true = None, None

    with torch.no_grad():
        n = 0
        for images, target in tqdm(dataloader):
            if (max_samples > 0) and (n >= max_samples):
                break
            images = images.to(device)
            target = target.to(device)
            n += images.shape[0]

            # Apply purification if enabled
            if use_purify:
                # Temporarily disable no_grad for purification
                with torch.enable_grad():
                    images = purify_images(
                        images,
                        _forward_unnorm,
                        alpha=purify_alpha,
                        eps=purify_eps,
                        t=purify_t,
                        max_iters=purify_max_iters
                    )

            with autocast():
                logits = _forward_unnorm(images)

            clean_true.append(target.cpu())
            clean_pred.append(logits.float().cpu())

    clean_pred = torch.cat(clean_pred)
    clean_true = torch.cat(clean_true)
    if max_samples > 0:
        clean_pred = clean_pred[:max_samples]
        clean_true = clean_true[:max_samples]
    print(f'[n samples] {len(clean_pred)}')
    return clean_pred, clean_true, pred, true

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap


def evaluate(model, dataloader, tokenizer, classnames, templates, device, normalize=None, resize=None,
             amp=True, verbose=False, save_clf=None, load_clfs=[], attack_config=None, purify_config=None):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    normalize: normalization transform

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    assert normalize is not None

    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    logits, target, adv_logits, adv_target = run_classification(model, classifier, dataloader, device,
                                        normalize=normalize, resize=resize, amp=amp,
                                        attack_config=attack_config, purify_config=purify_config)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        clean_acc, = accuracy(logits, target, topk=(1,))

        if adv_logits is None:
            adv_acc = float("nan") 
        else:
            adv_acc, = accuracy(adv_logits, adv_target, topk=(1,))

    
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        print(f"[clean_accuracy] {clean_acc*100:.2f}")
        print(f"[adv_accuracy] {adv_acc*100:.2f}")
        return {"clean_accuracy": clean_acc, "adv_accuracy": adv_acc, "mean_per_class_recall": mean_per_class_recall}
