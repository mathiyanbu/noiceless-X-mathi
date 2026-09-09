import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    
    SI-SNR(s_hat, s) = 10 * log10( ||s_target||^2 / ||e_noise||^2 )
      where s_target = ( <s_hat, s> / ||s||^2 ) * s
            e_noise  = s_hat - s_target
    """
    def __init__(self, eps: float = 1e-8, zero_mean: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean

    def forward(self, s_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_hat: Estimated waveform (B, T) or (B, 1, T)
            s: Clean target waveform (B, T) or (B, 1, T)
        Returns:
            Scalar negative SI-SNR loss (to be minimized)
        """
        if s_hat.dim() == 3:
            s_hat = s_hat.squeeze(1)
        if s.dim() == 3:
            s = s.squeeze(1)

        if self.zero_mean:
            s_hat = s_hat - torch.mean(s_hat, dim=-1, keepdim=True)
            s = s - torch.mean(s, dim=-1, keepdim=True)

        # Dot product <s_hat, s>
        dot = torch.sum(s_hat * s, dim=-1, keepdim=True)
        # Target signal energy ||s||^2
        s_energy = torch.sum(s * s, dim=-1, keepdim=True) + self.eps

        # Orthogonal projection: s_target = ( <s_hat, s> / ||s||^2 ) * s
        s_target = (dot / s_energy) * s
        e_noise = s_hat - s_target

        target_norm = torch.sum(s_target * s_target, dim=-1) + self.eps
        noise_norm = torch.sum(e_noise * e_noise, dim=-1) + self.eps

        si_snr = 10.0 * torch.log10(target_norm / noise_norm)
        # Negative SI-SNR so that minimizing loss maximizes SNR
        return -torch.mean(si_snr)


class STFTMagnitudeLoss(nn.Module):
    """
    Magnitude loss between |S_hat| and |S_clean| in the STFT domain.
    """
    def __init__(self, loss_type: str = "l1", eps: float = 1e-8):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps

    def forward(self, s_hat_complex: torch.Tensor, s_clean_complex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_hat_complex: (B, 2, T, F) where 2 = (Real, Imag)
            s_clean_complex: (B, 2, T, F)
        """
        mag_hat = torch.sqrt(s_hat_complex[:, 0]**2 + s_hat_complex[:, 1]**2 + self.eps)
        mag_clean = torch.sqrt(s_clean_complex[:, 0]**2 + s_clean_complex[:, 1]**2 + self.eps)

        if self.loss_type == "l2":
            return F.mse_loss(mag_hat, mag_clean)
        return F.l1_loss(mag_hat, mag_clean)


class ComplexSpectralLoss(nn.Module):
    """
    Direct L1/L2 loss on real and imaginary spectral components:
    L_complex = ||Re(S_hat) - Re(S_clean)|| + ||Im(S_hat) - Im(S_clean)||
    """
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type.lower()

    def forward(self, s_hat_complex: torch.Tensor, s_clean_complex: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            loss_r = F.mse_loss(s_hat_complex[:, 0], s_clean_complex[:, 0])
            loss_i = F.mse_loss(s_hat_complex[:, 1], s_clean_complex[:, 1])
        else:
            loss_r = F.l1_loss(s_hat_complex[:, 0], s_clean_complex[:, 0])
            loss_i = F.l1_loss(s_hat_complex[:, 1], s_clean_complex[:, 1])

        return loss_r + loss_i


class CompositeEnhancementLoss(nn.Module):
    """
    Combined multi-objective loss for complex speech enhancement:
      L = lambda_sisnr * L_SI-SNR + lambda_stft * L_STFT + lambda_complex * L_complex
    """
    def __init__(
        self,
        lambda_sisnr: float = 1.0,
        lambda_stft: float = 1.0,
        lambda_complex: float = 1.0,
        fft_size: int = 512,
        hop_size: int = 80
    ):
        super().__init__()
        self.lambda_sisnr = lambda_sisnr
        self.lambda_stft = lambda_stft
        self.lambda_complex = lambda_complex
        self.fft_size = fft_size
        self.hop_size = hop_size

        self.si_snr_loss = SISNRLoss()
        self.stft_loss = STFTMagnitudeLoss(loss_type="l1")
        self.complex_loss = ComplexSpectralLoss(loss_type="l1")

    def forward(
        self,
        s_hat_complex: torch.Tensor,
        s_clean_complex: torch.Tensor,
        s_hat_waveform: Optional[torch.Tensor] = None,
        s_clean_waveform: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted composite loss.
        """
        loss_stft = self.stft_loss(s_hat_complex, s_clean_complex)
        loss_cplx = self.complex_loss(s_hat_complex, s_clean_complex)

        total_loss = self.lambda_stft * loss_stft + self.lambda_complex * loss_cplx
        metrics = {
            "loss_stft": loss_stft.item(),
            "loss_complex": loss_cplx.item(),
        }

        if s_hat_waveform is not None and s_clean_waveform is not None and self.lambda_sisnr > 0.0:
            loss_sisnr = self.si_snr_loss(s_hat_waveform, s_clean_waveform)
            total_loss = total_loss + self.lambda_sisnr * loss_sisnr
            metrics["loss_sisnr"] = loss_sisnr.item()

        metrics["loss_total"] = total_loss.item()
        return total_loss, metrics
