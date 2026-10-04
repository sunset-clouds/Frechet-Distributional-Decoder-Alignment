import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FDloss.fd_loss import FrechetLoss
from models.discriminators import DinoDiscriminator
from models.perceptual_loss import PerceptualLoss
from utils.diff_aug import DiffAugment
from utils.util import Pack

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)

def hinge_gen_loss(logits_fake):
    return -torch.mean(logits_fake)

def adopt_weight(weight, cur_epoch, threshold=0, value=0.0):
    return value if cur_epoch < threshold else weight

class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)

def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
            torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg

class TrainingLoss(nn.Module):
    def __init__(self, args):
        super(TrainingLoss, self).__init__()
        self.args = args
        self.fd_loss = FrechetLoss(args.extractor_name, args.fd_loss_type, args.fid_stats_dir,
                                   weights=getattr(args, "fd_extractor_weights", None),
                                   norm_eps=getattr(args, "fd_norm_eps", 0.01),
                                   ema_beta=getattr(args, "fd_ema_beta", None),
                                   queue_size=getattr(args, "fd_queue_size", None)) if args.fd_weight > 0 else None
        self.perceptual_loss = PerceptualLoss(model_name="lpips-convnext_s-1.0-0.1").eval() if args.perceptual_weight > 0 else None
        self.discriminator = DinoDiscriminator() if args.disc_weight > 0 else None
        self.lecam_ema = LeCAM_EMA() if args.disc_weight > 0 else None
        self.disc_adaptive_weight = True if args.disc_weight > 0 else None
        self.args = args

    @property
    def has_discriminator(self):
        return self.discriminator is not None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, inputs, reconstructions, optimizer_idx, cur_epoch, last_layer=None,
                generated=None):
        """``reconstructions``: decode of encoder latents, scored by rec/perceptual/GAN against
        ``inputs``. ``generated``: decode of generation-time latents, scored by the FD loss;
        None puts the FD term on the reconstructions. ``reconstructions`` may be None when only
        the FD term is active. -> (loss, Pack of logged terms)."""
        disc_active = self.has_discriminator and cur_epoch >= self.args.disc_start_epoch
        if reconstructions is None:
            if self.args.rec_weight != 0 or self.perceptual_loss is not None or disc_active:
                raise ValueError(
                    "reconstructions=None requires rec_weight=0, perceptual_weight=0 and no "
                    "discriminator; got rec_weight="
                    f"{self.args.rec_weight}, perceptual={self.perceptual_loss is not None}, "
                    f"disc_active={disc_active}"
                )
            if generated is None:
                raise ValueError("reconstructions and generated cannot both be None")
        ref = reconstructions if reconstructions is not None else generated

        # decoder update
        if optimizer_idx == 0:
            if self.args.rec_weight !=0:
                rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous())
            else:
                rec_loss = torch.tensor(0.0, device=ref.device)

            if self.perceptual_loss is not None:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                p_loss = torch.mean(p_loss)
            else:
                p_loss = torch.tensor(0.0, device=ref.device)

            if self.fd_loss is not None:
                fd_source = reconstructions if generated is None else generated
                fd_loss = self.fd_loss(fd_source.contiguous())
            else:
                fd_loss = torch.tensor(0.0, device=ref.device)

            gen_loss = rec_loss * 0.0
            if self.args.rec_weight != 0:
                gen_loss = gen_loss + self.args.rec_weight * rec_loss
            if self.args.perceptual_weight != 0:
                gen_loss = gen_loss + self.args.perceptual_weight * p_loss
            if self.args.fd_weight != 0:
                gen_loss = gen_loss + self.args.fd_weight * fd_loss

            if not disc_active: ## not use the GAN
                loss_pack = Pack(gen_loss=gen_loss, rec_loss=rec_loss, lpips_loss=p_loss, fd_loss=fd_loss)
                if self.fd_loss is not None:
                    loss_pack.add(self.fd_loss.raw_log())
                return gen_loss, loss_pack
            else: ## use GAN
                reconstructions_aug = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
                logits_fake = self.discriminator(reconstructions_aug.contiguous())
                g_loss = hinge_gen_loss(logits_fake)

                if self.disc_adaptive_weight:
                    null_loss = self.args.rec_weight * rec_loss + self.args.perceptual_weight * p_loss
                    disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, g_loss, last_layer=last_layer)
                else:
                    disc_adaptive_weight = 1. 

                disc_weight = adopt_weight(self.args.disc_weight, cur_epoch, threshold=self.args.disc_start_epoch)
                gen_loss = gen_loss + disc_adaptive_weight * disc_weight * g_loss 
        
                loss_pack = Pack(gen_loss=gen_loss, rec_loss=rec_loss, lpips_loss=p_loss, fd_loss=fd_loss, g_loss=g_loss, disc_weight=disc_weight, disc_adaptive_weight=disc_adaptive_weight)
                if self.fd_loss is not None:
                    loss_pack.add(self.fd_loss.raw_log())
                return gen_loss, loss_pack

        # discriminator update
        if optimizer_idx == 1:
            if not disc_active:
                d_loss = torch.tensor(0.0, device=inputs.device)
                loss_pack = Pack(d_loss=d_loss)
                return d_loss, loss_pack
            else:
                logits_real = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
                logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
                disc_weight = adopt_weight(self.args.disc_weight, cur_epoch, threshold=self.args.disc_start_epoch)

                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                adversarial_loss = hinge_d_loss(logits_real, logits_fake)
                d_loss = disc_weight * (lecam_loss * self.args.lecam_loss_weight + adversarial_loss)

                logits_real_s = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                disc_cr_loss_weight = self.args.disc_cr_loss_weight if cur_epoch >= self.args.disc_start_epoch else 0.0
                d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
                d_loss += d_cr

                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                loss_pack = Pack(d_loss=d_loss, logits_real=logits_real, logits_fake=logits_fake)
                return d_loss, loss_pack

   