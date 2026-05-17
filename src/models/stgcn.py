"""
STGCN — Spatio-Temporal Graph Convolutional Network (Yu et al., 2018).
Architecture: ST-Conv Block (Temporal Conv → Graph Conv → Temporal Conv) × 2 → Output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution."""

    def __init__(self, K, in_channels, out_channels):
        """
        Args:
            K: Chebyshev polynomial order.
            in_channels: Input feature channels.
            out_channels: Output feature channels.
        """
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K + 1, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, cheb_polys):
        """
        Args:
            x: (batch, N, in_channels)
            cheb_polys: list of K+1 Chebyshev polynomial matrices, each (N, N)
        Returns:
            (batch, N, out_channels)
        """
        batch, N, C_in = x.shape
        outputs = []
        for k in range(self.K + 1):
            # T_k(L) @ x: (N, N) @ (batch, N, C_in) → need batched matmul
            Tk = cheb_polys[k]  # (N, N)
            Tk_x = torch.einsum('mn,bnc->bmc', Tk, x)  # (batch, N, C_in)
            outputs.append(Tk_x)

        # Stack and multiply by weights
        # outputs: list of K+1 tensors, each (batch, N, C_in)
        x_cheb = torch.stack(outputs, dim=0)  # (K+1, batch, N, C_in)
        # weight: (K+1, C_in, C_out)
        out = torch.einsum('kbnc,kco->bno', x_cheb, self.weight)  # (batch, N, C_out)
        return out + self.bias


class TemporalConv(nn.Module):
    """Gated temporal convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 2 * out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size - 1) // 2),
        )
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: (batch, C_in, N, T)
        Returns:
            (batch, C_out, N, T)
        """
        conv_out = self.conv(x)  # (batch, 2*C_out, N, T)
        P, Q = conv_out.split(self.out_channels, dim=1)
        return P * torch.sigmoid(Q)  # GLU gating


class STConvBlock(nn.Module):
    """Spatio-Temporal Convolution Block: Temporal → Spatial → Temporal."""

    def __init__(self, K, in_channels, spatial_channels, out_channels,
                 kernel_size=3):
        super().__init__()
        self.temporal1 = TemporalConv(in_channels, spatial_channels, kernel_size)
        self.graph_conv = ChebGraphConv(K, spatial_channels, spatial_channels)
        self.temporal2 = TemporalConv(spatial_channels, out_channels, kernel_size)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, cheb_polys):
        """
        Args:
            x: (batch, C_in, N, T)
            cheb_polys: list of Chebyshev polynomial tensors on device
        Returns:
            (batch, C_out, N, T)
        """
        # Temporal conv 1
        t1 = self.temporal1(x)  # (batch, spatial_ch, N, T)

        # Graph conv (need to reshape for per-timestep graph conv)
        batch, C, N, T = t1.shape
        # Reshape to (batch*T, N, C) for graph conv
        t1_reshaped = t1.permute(0, 3, 2, 1).reshape(batch * T, N, C)
        gc = self.graph_conv(t1_reshaped, cheb_polys)  # (batch*T, N, C)
        gc = gc.reshape(batch, T, N, C).permute(0, 3, 2, 1)  # (batch, C, N, T)
        gc = F.relu(gc)

        # Temporal conv 2
        t2 = self.temporal2(gc)  # (batch, C_out, N, T)

        # Layer norm (over channel dimension)
        t2 = t2.permute(0, 2, 3, 1)  # (batch, N, T, C_out)
        t2 = self.layer_norm(t2)
        t2 = self.dropout(t2)
        t2 = t2.permute(0, 3, 1, 2)  # (batch, C_out, N, T)

        return t2


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Two ST-Conv blocks followed by output temporal convolution.
    """

    def __init__(self, num_sensors, seq_len=12, pred_len=12, K=3,
                 channels=None):
        """
        Args:
            num_sensors: Number of sensor nodes N.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            K: Chebyshev polynomial order.
            channels: List of channel dimensions [in, mid1, mid2, out].
        """
        super().__init__()
        if channels is None:
            channels = [1, 16, 32, 64]

        self.num_sensors = num_sensors
        self.pred_len = pred_len

        self.block1 = STConvBlock(K, channels[0], channels[1], channels[2])
        self.block2 = STConvBlock(K, channels[2], channels[2], channels[3])

        # Output layer
        self.output_conv = nn.Conv2d(channels[3], pred_len, kernel_size=(1, seq_len))
        self.output_linear = nn.Linear(num_sensors, num_sensors)

    def forward(self, x, cheb_polys):
        """
        Args:
            x: (batch, seq_len, num_sensors)
            cheb_polys: list of Chebyshev polynomial tensors on device
        Returns:
            (batch, pred_len, num_sensors)
        """
        batch = x.size(0)

        # Reshape to (batch, 1, N, T) — channels=1, spatial=N, temporal=T
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, N, T)

        # ST-Conv blocks
        x = self.block1(x, cheb_polys)  # (batch, 32, N, T)
        x = self.block2(x, cheb_polys)  # (batch, 64, N, T)

        # Output
        x = self.output_conv(x)  # (batch, pred_len, N, 1)
        x = x.squeeze(-1)  # (batch, pred_len, N)
        x = self.output_linear(x)  # (batch, pred_len, N)

        return x

    def get_name(self):
        return "STGCN"
