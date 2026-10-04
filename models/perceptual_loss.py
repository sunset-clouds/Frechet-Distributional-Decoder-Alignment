"""Perceptual loss from LPIPS and/or ConvNeXt-S features."""
import torch

from torchvision import models
from models.lpips import LPIPS

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

 
class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "lpips"):
        """model_name: "lpips", "convnext_s", or "lpips-convnext_s-{w_lpips}-{w_convnext}", giving
        loss = (w_lpips * lpips + w_convnext * convnext) / (w_lpips + w_convnext)."""
        super().__init__()
        if ("lpips" not in model_name) and (
            "convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        if "lpips" in model_name:
            self.lpips = LPIPS().eval()

        if "convnext_s" in model_name:
            self.convnext = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split('-')[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(loss_config[0]), float(loss_config[1])
            print(f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}")
 
        self.register_buffer("imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """input, target: (B, 3, H, W) in [-1, 1] -> (B, 1, 1, 1) with LPIPS, scalar otherwise."""
        self.eval()
        loss = 0.
        num_losses = 0.
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss

        if self.convnext is not None:
            pred_input = torch.nn.functional.interpolate(input, size=224, mode="bicubic", align_corners=False, antialias=True)
            pred_target = torch.nn.functional.interpolate(target, size=224, mode="bicubic", align_corners=False, antialias=True)

            # shift from [-1, 1] to [0, 1]
            pred_input = (pred_input + 1.) / 2.
            pred_target = (pred_target + 1.) / 2.

            pred_input = self.convnext((pred_input - self.imagenet_mean) / self.imagenet_std)
            pred_target = self.convnext((pred_target - self.imagenet_mean) / self.imagenet_std)
            convnext_loss = torch.nn.functional.mse_loss(
                pred_input,
                pred_target,
                reduction="mean")
                
            if self.loss_weight_convnext is None:
                num_losses += 1
                loss += convnext_loss
            else:
                num_losses += self.loss_weight_convnext
                loss += self.loss_weight_convnext * convnext_loss

        loss = loss / num_losses
        return loss