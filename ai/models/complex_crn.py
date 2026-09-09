import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ComplexCRN(nn.Module):
    """
    Complex-Domain Convolutional Recurrent Network (CRN) for Speech Enhancement.
    
    Architecture:
    - Complex Encoder: 2D frequency-axis convolutions compressing frequency bins (256 -> 128 -> 64 -> 32).
    - Recurrent Block: 2-layer causal GRU modeling temporal evolution across frames with persistent hidden states.
    - Complex Decoder: Mirror transpose convolutions with U-Net skip connections.
    - Complex Ratio Mask (CRM):
        M_hat = M_R + j * M_I
        S_hat = M_hat * X (complex multiplication with noisy STFT X)
    - Streaming Inference:
        step(frame, hidden_state) -> (enhanced_frame, new_hidden_state)
    """
    def __init__(
        self,
        num_bins: int = 257,
        encoder_channels: Tuple[int, ...] = (16, 32, 64),
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        mask_bound: float = 2.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mask_bound = mask_bound
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # 1. Complex Encoder (Operates over (Real, Imag) channels along frequency axis)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, encoder_channels[0], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.PReLU(encoder_channels[0])
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.PReLU(encoder_channels[1])
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(encoder_channels[2]),
            nn.PReLU(encoder_channels[2])
        )

        # Bottleneck feature dimension: 64 channels * 32 frequency bins = 2048
        self.bottleneck_dim = encoder_channels[2] * 32
        self.proj_in = nn.Linear(self.bottleneck_dim, gru_hidden_size)

        # 2. Causal Temporal Recurrent Block
        self.gru = nn.GRU(
            input_size=gru_hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )
        self.proj_out = nn.Linear(gru_hidden_size, self.bottleneck_dim)

        # 3. Complex Decoder with U-Net Skip Connections
        # Skip 3: dec3_in has encoder_channels[2] + encoder_channels[2] = 128
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[2] * 2, encoder_channels[1], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1)),
            nn.BatchNorm2d(encoder_channels[1]),
            nn.PReLU(encoder_channels[1])
        )
        # Skip 2: dec2_in has encoder_channels[1] + encoder_channels[1] = 64
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[1] * 2, encoder_channels[0], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1)),
            nn.BatchNorm2d(encoder_channels[0]),
            nn.PReLU(encoder_channels[0])
        )
        # Skip 1: dec1_in has encoder_channels[0] + encoder_channels[0] = 32 -> Output 2 channels (M_R, M_I)
        self.dec1 = nn.ConvTranspose2d(encoder_channels[0] * 2, 2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1))

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize zero hidden state for streaming GRU inference."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(self.gru_num_layers, batch_size, self.gru_hidden_size, device=dev)

    def _apply_mask(self, x_complex: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication: S_hat = M * X
        M_hat = M_R + j * M_I
        X     = X_R + j * X_I
        S_hat_R = M_R * X_R - M_I * X_I
        S_hat_I = M_R * X_I + M_I * X_R
        """
        m_r = mask[:, 0]
        m_i = mask[:, 1]
        x_r = x_complex[:, 0]
        x_i = x_complex[:, 1]

        s_hat_r = m_r * x_r - m_i * x_i
        s_hat_i = m_r * x_i + m_i * x_r

        return torch.stack([s_hat_r, s_hat_i], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch forward execution for training and evaluation.
        Args:
            x: Input noisy STFT tensor of shape (B, 2, T, F) where 2 = (Real, Imag)
            hidden: Optional initial hidden state for GRU of shape (num_layers, B, hidden_dim)
        Returns:
            s_hat: Enhanced complex STFT of shape (B, 2, T, F)
            mask: Estimated complex ratio mask of shape (B, 2, T, F)
            new_hidden: Updated GRU hidden state
        """
        b, c, t, f = x.shape
        if not torch.jit.is_tracing():
            assert c == 2, f"Expected 2 complex channels (Real, Imag), got {c}"

        # Align 257 bins to 256 for power-of-2 downsampling
        if f == 257:
            x_enc_in = x[..., :256]
        else:
            x_enc_in = x

        # Encoder forward
        e1 = self.enc1(x_enc_in)  # (B, 16, T, 128)
        e2 = self.enc2(e1)        # (B, 32, T, 64)
        e3 = self.enc3(e2)        # (B, 64, T, 32)

        # Reshape for GRU: (B, 64, T, 32) -> (B, T, 64, 32) -> (B, T, 2048)
        e3_perm = e3.permute(0, 2, 1, 3).contiguous()
        gru_in_flat = e3_perm.view(b, t, -1)
        gru_in = self.proj_in(gru_in_flat)

        # Temporal GRU
        if hidden is None:
            hidden = self.init_hidden(b, device=x.device)
        gru_out, new_hidden = self.gru(gru_in, hidden)

        gru_out_proj = self.proj_out(gru_out)
        d3_in = gru_out_proj.view(b, t, 64, 32).permute(0, 2, 1, 3).contiguous()

        # Decoder forward with U-Net skip connections
        d3 = self.dec3(torch.cat([d3_in, e3], dim=1))  # (B, 32, T, 64)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))     # (B, 16, T, 128)
        raw_mask_256 = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 2, T, 256)

        # Pad Nyquist bin back if input had 257 bins
        if f == 257:
            mask_raw = torch.cat([raw_mask_256, raw_mask_256[..., -1:]], dim=-1)
        else:
            mask_raw = raw_mask_256

        # Bounded Complex Ratio Mask
        mask = torch.tanh(mask_raw) * self.mask_bound

        # Complex Multiplication: S_hat = M * X
        s_hat = self._apply_mask(x, mask)

        return s_hat, mask, new_hidden

    def step(
        self,
        frame: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Streaming single-frame inference interface for real-time DSP pipeline.
        Args:
            frame: Single-frame noisy STFT tensor of shape (B, 2, F) or (B, 2, 1, F)
            hidden: Current GRU hidden state of shape (num_layers, B, hidden_dim)
        Returns:
            enhanced_frame: Enhanced STFT frame of matching shape
            mask: Computed complex ratio mask
            new_hidden: Updated GRU hidden state
        """
        was_3d = False
        if frame.dim() == 3:
            frame = frame.unsqueeze(2)  # (B, 2, 1, F)
            was_3d = True

        s_hat, mask, new_hidden = self.forward(frame, hidden)

        if was_3d:
            s_hat = s_hat.squeeze(2)
            mask = mask.squeeze(2)

        return s_hat, mask, new_hidden
