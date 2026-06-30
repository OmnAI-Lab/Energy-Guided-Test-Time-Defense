import torch

def energy_x(logits, t=1.0):
    """Energy function for purification (supporta [B,V] o [B,L,V])."""
    return -t * torch.logsumexp(logits / t, dim=-1)  # [B]

imagenet_features = torch.load("imagenet21k_words/embeddings/vitl14-openai/text_embeddings.pt").to("cuda", dtype=torch.float32)


def _clip_feats_from_llava(eval_model, x_query):
    """
    Estrae l'embedding CLIP (CLS token normalizzato) da un modello LLaVA
    che monta un vision tower compatibile con forward_full_clip().
    """
    model = eval_model.model

    if not hasattr(model, "get_vision_tower"):
        raise RuntimeError("Impossibile accedere al vision tower: manca get_vision_tower()")

    vt = model.get_vision_tower()
    if vt is None:
        raise RuntimeError("Vision tower restituito da get_vision_tower() è None")

    if not hasattr(vt, "forward_full_clip"):
        raise RuntimeError("La vision tower non implementa forward_full_clip().")

    print(f"x_query shape: {x_query.shape}")

    feats = vt.full_clip.encode_image(x_query)  # [B, D] — già normalizzati

    print(f"Final CLIP features shape: {feats.shape}")
    
    #todo chiedi hussain
    
    feats = feats.to("cuda", dtype=torch.float32)
    
    logits = 100. * feats @ imagenet_features.T
    print(f"first 20 logits for first sample: {logits[0,:20]}")

    return logits




# def purify_images(images, model_forward_fn, alpha=2.5, eps=5, t=1.0, max_iters=2):
#     """
#     Purify images using energy-based refinement.
    
#     Parameters
#     ----------
#     images: torch.Tensor
#         Input images to purify (B, C, H, W)
#     model_forward_fn: callable
#         Function that takes images and returns logits
#     alpha: float
#         Step size for purification
#     eps: float
#         Maximum L2 perturbation budget
#     t: float
#         Temperature parameter for energy function
#     max_iters: int
#         Maximum purification iterations
    
#     Returns
#     -------
#     torch.Tensor
#         Purified images with same shape as input
#     """
    
#     print(f"Input images size: {images.shape}")
    
#     device = images.device
#     dtype = images.dtype
    
#     with torch.enable_grad():
#         purified_x = images.clone().detach()#.float()
#         purified_x.requires_grad = True
#         perturbation_norm = torch.zeros(images.size(0), 1, 1, 1, device=device, dtype=dtype)
#         for i in range(max_iters):
#             # alpha = alpha / (i + 1)  # decay step size
#             # Early exit if all samples reached eps
#             if (perturbation_norm >= eps).all():
#                 break
                        
#             print(f">> purified_x range: {purified_x.min().item()} - {purified_x.max().item()}")
#             logits = model_forward_fn(purified_x)
#             print(f">> logits stats: min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")

#             energy_vec = energy_x(logits, t=t)
#             if torch.isnan(energy_vec).any():
#                 print("NaN in energy!")
#                 break

#             grad = torch.autograd.grad(energy_vec.sum(), purified_x, create_graph=False)[0]

#             print(f">> grad stats: min={grad.min().item()}, max={grad.max().item()}, mean={grad.mean().item()}")

#             if torch.isnan(grad).any():
#                 print("NaN in grad!")
#                 break

#             grad = grad.to(dtype)  # <<< importantissimo: allinea tipo >>>


#             # Compute per-sample grad L2 and keep dims for broadcasting
#             grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)

#             # Mask out samples that already reached eps
#             active = (perturbation_norm < eps).float().to(dtype)
            
#             with torch.no_grad():
#                 purified_x = purified_x - alpha * (grad / grad_norm) * active
#                 purified_x = torch.clamp(purified_x, 0, 1)

#                 perturbation = purified_x - images
#                 # perturbation = torch.clamp(perturbation, -16/255, 16/255)
#                 perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), 
#                                               p=2, dim=1, keepdim=True).view(-1, 1, 1, 1)

#                 factor = torch.clamp(eps / (perturbation_norm + 1e-3), max=1.0)
#                 perturbation = factor * perturbation

#                 purified_x = torch.clamp(images + perturbation, 0, 1)
#                 purified_x = purified_x.detach().to(device=device, dtype=dtype)
            
#             purified_x.requires_grad = True


#     print ("images purified after {} iters".format(i+1))
#     print(f"Purified image size: {purified_x.shape}")
    
#     print(f"purified image max value: {purified_x.max().item()}, min value: {purified_x.min().item()}")
    
#     return purified_x.detach().to(device=device, dtype=dtype)


def purify_images(images, model_forward_fn, alpha=1, eps=4, t=1.0, max_iters=10):
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
        Maximum L-infinity perturbation budget
    t: float
        Temperature parameter for energy function
    max_iters: int
        Maximum purification iterations
    
    Returns
    -------
    torch.Tensor
        Purified images with same shape as input
    """
    eps = eps / 255.0  # convert to [0,1] scale
    with torch.enable_grad():
        purified_x = images.clone().detach()
        purified_x.requires_grad = True
        
        for i in range(max_iters):
            logits = model_forward_fn(purified_x)
            energy_vec = energy_x(logits, t=t)
            grad = torch.autograd.grad(energy_vec.sum(), purified_x, create_graph=False)[0]

            # Normalize gradient using L2 norm for step direction
            grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            
            with torch.no_grad():
                # Take step in gradient direction
                purified_x = purified_x - alpha * (grad / grad_norm)
                purified_x = torch.clamp(purified_x, 0, 1)

                # Compute perturbation and project to L-infinity ball
                perturbation = purified_x - images
                perturbation = torch.clamp(perturbation, -eps, eps)

                # Apply clamped perturbation
                purified_x = torch.clamp(images + perturbation, 0, 1)
                purified_x = purified_x.detach()
            
            purified_x.requires_grad = True

    return purified_x.detach()