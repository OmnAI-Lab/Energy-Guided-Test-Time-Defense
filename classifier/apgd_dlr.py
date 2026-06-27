''' The code has been adapted from the original implementation of Transfer APGD-DLR loss + BPDA. The original implementation can be found at:
https://github.com/fra31/evaluating-adaptive-test-time-defenses '''

import math
from functools import partial

import torch
import torch.nn.functional as F


def l2_norm(x, keepdim=False):
    shape = x.shape
    return torch.norm(
        x.reshape(shape[0], -1),
        p=2,
        dim=1,
        keepdim=keepdim,
    ).view(-1, *([1] * (len(shape) - 1))) if keepdim else torch.norm(
        x.reshape(shape[0], -1),
        p=2,
        dim=1,
    )


def dlr_loss(logits, y):
    x_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    numerator = logits[torch.arange(logits.shape[0], device=logits.device), y]
    numerator = numerator - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)

    denominator = x_sorted[:, -1] - x_sorted[:, -3] + 1e-12

    return -numerator / denominator


def dlr_loss_targeted(logits, y, y_target):
    x_sorted, _ = logits.sort(dim=1)
    u = torch.arange(logits.shape[0], device=logits.device)

    numerator = logits[u, y] - logits[u, y_target]
    denominator = x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12

    return -numerator / denominator


def cw_loss(logits, y):
    x_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    numerator = logits[torch.arange(logits.shape[0], device=logits.device), y]
    numerator = numerator - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)

    return -numerator


def ce_loss(logits, y):
    return F.cross_entropy(logits, y, reduction="none")


def get_loss_fn(loss_name, y_target=None):
    if loss_name == "ce":
        return ce_loss

    if loss_name == "dlr":
        return dlr_loss

    if loss_name == "cw":
        return cw_loss

    if loss_name == "dlr-targeted":
        if y_target is None:
            raise ValueError("y_target is required for dlr-targeted loss.")
        return partial(dlr_loss_targeted, y_target=y_target)

    raise ValueError(f"Unsupported loss: {loss_name}")


def check_oscillation(loss_steps, j, k, k3=0.75):
    """APGD oscillation check."""
    device = loss_steps.device
    t = torch.zeros(loss_steps.shape[1], device=device)

    for counter in range(k):
        t += (loss_steps[j - counter] > loss_steps[j - counter - 1]).float()

    return (t <= k * k3).float()


def apgd_single_run(
    model,
    x,
    y,
    *,
    norm="Linf",
    eps=4.0 / 255.0,
    n_iter=100,
    use_random_start=True,
    loss="dlr-targeted",
    y_target=None,
    early_stop=True,
    eot_iter=0,
):
    """
    Single APGD run with adaptive step size, random start,
    and Linf/L2 projection.
    """
    assert not model.training

    device = x.device
    ndims = len(x.shape) - 1

    if use_random_start:
        if norm == "Linf":
            t = (torch.rand_like(x) - 0.5) * 2.0 * eps
            x_adv = (x + t).clamp(0.0, 1.0)
        elif norm == "L2":
            t = torch.randn_like(x)
            t_norm = l2_norm(t, keepdim=True)
            r = torch.rand(
                x.shape[0],
                *([1] * ndims),
                device=device,
            )
            t = t / (t_norm + 1e-12) * r * eps
            x_adv = (x + t).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    else:
        x_adv = x.clone()

    x_adv = x_adv.clamp(0.0, 1.0)

    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()

    loss_steps = torch.zeros((n_iter, x.shape[0]), device=device)
    loss_best_steps = torch.zeros((n_iter + 1, x.shape[0]), device=device)
    acc_steps = torch.zeros((n_iter + 1, x.shape[0]), device=device)

    loss_adv = -float("inf") * torch.ones(x.shape[0], device=device)

    criterion_indiv = get_loss_fn(loss, y_target=y_target)

    if norm in ["Linf", "L2"]:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)

        k = n_iter_2
        threshold_decrease = 0.75
        alpha = 2.0
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    step_size = alpha * eps * torch.ones(
        x.shape[0],
        *([1] * ndims),
        device=device,
    )

    counter3 = 0

    # Initial gradient.
    x_adv.requires_grad_()
    logits = model(x_adv)
    loss_indiv = criterion_indiv(logits, y)
    loss_sum = loss_indiv.sum()

    grad = torch.autograd.grad(loss_sum, [x_adv])[0].detach()

    for _ in range(eot_iter):
        loss_indiv_curr = criterion_indiv(model(x_adv), y)
        grad += torch.autograd.grad(loss_indiv_curr.sum(), [x_adv])[0].detach()
        loss_indiv += loss_indiv_curr.detach()

    grad /= float(eot_iter + 1)

    grad_best = grad.clone()
    x_adv = x_adv.detach()

    acc = logits.detach().argmax(dim=1) == y
    acc_steps[0] = acc.float()

    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)

    loss_best_steps[0] = loss_best

    x_adv_old = x_adv.clone().detach()

    for i in range(n_iter):
        x_adv = x_adv.detach()

        grad2 = x_adv - x_adv_old
        x_adv_old = x_adv.clone()

        a = 0.75 if i > 0 else 1.0

        if norm == "Linf":
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.max(torch.min(x_adv_1, x + eps), x - eps)
            x_adv_1 = x_adv_1.clamp(0.0, 1.0)

            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1.0 - a)
            x_adv_1 = torch.max(torch.min(x_adv_1, x + eps), x - eps)
            x_adv_1 = x_adv_1.clamp(0.0, 1.0)

        elif norm == "L2":
            grad_norm = l2_norm(grad, keepdim=True)
            x_adv_1 = x_adv + step_size * grad / (grad_norm + 1e-12)

            d = x_adv_1 - x
            d_norm = l2_norm(d, keepdim=True)
            d = d / (d_norm + 1e-12) * torch.min(
                eps * torch.ones_like(d_norm),
                d_norm,
            )
            x_adv_1 = (x + d).clamp(0.0, 1.0)

            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1.0 - a)

            d = x_adv_1 - x
            d_norm = l2_norm(d, keepdim=True)
            d = d / (d_norm + 1e-12) * torch.min(
                eps * torch.ones_like(d_norm),
                d_norm,
            )
            x_adv_1 = (x + d).clamp(0.0, 1.0)

        x_adv = x_adv_1.detach()
        x_adv.requires_grad_()

        logits = model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        loss_sum = loss_indiv.sum()

        grad = torch.autograd.grad(loss_sum, [x_adv])[0].detach()

        for _ in range(eot_iter):
            loss_indiv_curr = criterion_indiv(model(x_adv), y)
            grad += torch.autograd.grad(loss_indiv_curr.sum(), [x_adv])[0].detach()
            loss_indiv += loss_indiv_curr.detach()

        grad /= float(eot_iter + 1)

        x_adv = x_adv.detach()
        loss_indiv = loss_indiv.detach()

        pred_correct = logits.detach().argmax(dim=1) == y

        acc_old = acc.clone()
        acc = torch.logical_and(acc, pred_correct)
        acc_steps[i + 1] = acc.float()

        newly_fooled = torch.logical_and(~pred_correct, acc_old)
        better_fooled = torch.logical_and(~pred_correct, loss_indiv > loss_adv)
        update_adv = torch.logical_or(newly_fooled, better_fooled)

        if update_adv.any():
            x_best_adv[update_adv] = x_adv[update_adv].clone()
            loss_adv[update_adv] = loss_indiv[update_adv].clone()

        loss_steps[i] = loss_indiv

        better_loss = loss_indiv > loss_best
        if better_loss.any():
            x_best[better_loss] = x_adv[better_loss].clone()
            grad_best[better_loss] = grad[better_loss].clone()
            loss_best[better_loss] = loss_indiv[better_loss].clone()

        loss_best_steps[i + 1] = loss_best

        counter3 += 1

        if counter3 == k:
            oscillation = check_oscillation(
                loss_steps,
                i,
                k,
                k3=threshold_decrease,
            )

            no_improvement = (1.0 - reduced_last_check) * (
                loss_best_last_check >= loss_best
            ).float()

            oscillation = torch.max(oscillation, no_improvement)

            reduced_last_check = oscillation.clone()
            loss_best_last_check = loss_best.clone()

            reduce_mask = oscillation > 0

            if reduce_mask.any():
                step_size[reduce_mask] /= 2.0
                x_adv[reduce_mask] = x_best[reduce_mask].clone()
                grad[reduce_mask] = grad_best[reduce_mask].clone()

            counter3 = 0
            k = max(k - size_decr, n_iter_min)

        if early_stop and acc.sum() == 0:
            break

    return x_best, acc, loss_best, x_best_adv


def apgd_t_dlr_restarts(
    model,
    x,
    y,
    *,
    norm="Linf",
    eps=4.0 / 255.0,
    n_iter=100,
    n_restarts=5,
    eot_iter=0,
    early_stop=False,
    verbose=False,
):
    """
    Targeted APGD with DLR loss and multiple restarts.

    Returns:
        x_adv:
            Successful adversarial examples when found,
            original images for samples not fooled.

        x_best:
            Best-loss images for samples not fooled.
    """
    model.eval()

    device = x.device
    x_adv = x.clone()
    x_best = x.clone()

    robust_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
    loss_best = -float("inf") * torch.ones(x.shape[0], device=device)

    with torch.no_grad():
        output = model(x)
        output_sorted = output.sort(dim=1)[1]

    # Cycle through 9 target class ranks across restarts.
    n_target_classes = 9

    for restart_idx in range(n_restarts):
        if robust_mask.sum() == 0:
            break

        target_rank = restart_idx % n_target_classes + 2
        y_target_all = output_sorted[:, -target_rank]
        y_target = y_target_all[robust_mask]

        if verbose:
            print(f"Restart {restart_idx + 1}/{n_restarts}, target rank {target_rank}")

        x_curr = x[robust_mask]
        y_curr = y[robust_mask]

        x_best_curr, _, loss_curr, x_adv_curr = apgd_single_run(
            model=model,
            x=x_curr,
            y=y_curr,
            norm=norm,
            eps=eps,
            n_iter=n_iter,
            use_random_start=True,
            loss="dlr-targeted",
            y_target=y_target,
            early_stop=early_stop,
            eot_iter=eot_iter,
        )

        with torch.no_grad():
            pred_curr = model(x_adv_curr).argmax(dim=1)
            acc_curr = pred_curr == y_curr

        robust_indices = torch.nonzero(robust_mask, as_tuple=False).squeeze(1)

        fooled_local = ~acc_curr
        if fooled_local.any():
            fooled_global = robust_indices[fooled_local]
            x_adv[fooled_global] = x_adv_curr[fooled_local].clone()

        still_robust_local = acc_curr
        better_local = torch.logical_and(
            still_robust_local,
            loss_curr > loss_best[robust_indices],
        )

        if better_local.any():
            better_global = robust_indices[better_local]
            x_best[better_global] = x_best_curr[better_local].clone()
            loss_best[better_global] = loss_curr[better_local].clone()

        robust_mask[robust_indices] = acc_curr

        if verbose:
            robust_acc = 100.0 * robust_mask.float().mean().item()
            print(f"Restart {restart_idx + 1}: robust accuracy = {robust_acc:.2f}%")

    # For fooled samples, x_best becomes the successful adversarial.
    x_best[~robust_mask] = x_adv[~robust_mask].clone()

    return x_adv, x_best