import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    """
    Generate adversarial examples using FGSM on the given model.

    Args:
        model: torch.nn.Module
        images: input tensor [B x C x H x W]
        labels: ground truth labels
        epsilon: perturbation strength

    Returns:
        perturbed images (clamped between 0 and 1)
    """
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.data.sign()
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()
