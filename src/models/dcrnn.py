"""
DCRNN — Diffusion Convolutional Recurrent Neural Network (Li et al., 2018).
Architecture: Encoder-Decoder with Diffusion Conv GRU cells.
Spatial: Bidirectional random walk diffusion on graph.
Temporal: GRU-based sequence-to-sequence.
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionConv(nn.Module):
    """
    Diffusion convolution layer.
    Performs bidirectional random walk diffusion on the graph.
    """

    def __init__(self, num_supports, in_channels, out_channels):
        """
        Args:
            num_supports: Number of diffusion support matrices (2*K for bidirectional).
            in_channels: Input feature channels.
            out_channels: Output feature channels.
        """
        super().__init__()
        # +1 for the identity (self-connection)
        total_supports = num_supports + 1
        self.weight = nn.Parameter(
            torch.FloatTensor(total_supports, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.num_supports = num_supports

    def forward(self, x, supports):
        """
        Args:
            x: (batch, N, in_channels)
            supports: list of diffusion matrices, each (N, N)
        Returns:
            (batch, N, out_channels)
        """
        batch, N, C = x.shape

        # Identity convolution
        out = torch.einsum('bnc,co->bno', x, self.weight[0])

        # Diffusion convolutions
        for k, S in enumerate(supports):
            Sx = torch.einsum('mn,bnc->bmc', S, x)  # (batch, N, C)
            out = out + torch.einsum('bnc,co->bno', Sx, self.weight[k + 1])

        return out + self.bias


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    Replaces FC layers in standard GRU with diffusion convolution.
    """

    def __init__(self, num_supports, in_channels, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gates: reset and update (combined for efficiency)
        self.gate_conv = DiffusionConv(
            num_supports, in_channels + hidden_dim, 2 * hidden_dim
        )
        # Candidate
        self.candidate_conv = DiffusionConv(
            num_supports, in_channels + hidden_dim, hidden_dim
        )

    def forward(self, x, h, supports):
        """
        Args:
            x: (batch, N, in_channels) — input at current time step
            h: (batch, N, hidden_dim) — previous hidden state
            supports: list of diffusion matrices
        Returns:
            h_new: (batch, N, hidden_dim)
        """
        combined = torch.cat([x, h], dim=-1)  # (batch, N, in+hidden)

        gates = self.gate_conv(combined, supports)  # (batch, N, 2*hidden)
        gates = torch.sigmoid(gates)
        r, u = gates.split(self.hidden_dim, dim=-1)

        # Candidate
        combined_r = torch.cat([x, r * h], dim=-1)
        candidate = torch.tanh(self.candidate_conv(combined_r, supports))

        h_new = u * h + (1 - u) * candidate
        return h_new

    def init_hidden(self, batch_size, num_nodes, device):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)


class DCRNNEncoder(nn.Module):
    """DCRNN Encoder: stack of DCGRU cells."""

    def __init__(self, num_supports, in_channels, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(num_supports, in_channels, hidden_dim))
        for _ in range(1, num_layers):
            self.cells.append(DCGRUCell(num_supports, hidden_dim, hidden_dim))

    def forward(self, x_seq, supports):
        """
        Args:
            x_seq: (batch, seq_len, N, C_in)
            supports: list of diffusion matrices
        Returns:
            hidden_states: list of (batch, N, hidden_dim) for each layer
        """
        batch, seq_len, N, C = x_seq.shape
        device = x_seq.device

        # Initialize hidden states
        h_list = [cell.init_hidden(batch, N, device) for cell in self.cells]

        for t in range(seq_len):
            x_t = x_seq[:, t]  # (batch, N, C)
            for layer_idx, cell in enumerate(self.cells):
                inp = x_t if layer_idx == 0 else h_list[layer_idx - 1]
                h_list[layer_idx] = cell(inp, h_list[layer_idx], supports)

        return h_list


class DCRNNDecoder(nn.Module):
    """DCRNN Decoder: stack of DCGRU cells with projection."""

    def __init__(self, num_supports, hidden_dim, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(num_supports, out_channels, hidden_dim))
        for _ in range(1, num_layers):
            self.cells.append(DCGRUCell(num_supports, hidden_dim, hidden_dim))
        self.projection = nn.Linear(hidden_dim, out_channels)

    def forward(self, h_list, supports, pred_len, teacher_forcing=None,
                tf_ratio=0.0):
        """
        Args:
            h_list: Encoder hidden states (list of tensors)
            supports: Diffusion matrices
            pred_len: Number of steps to decode
            teacher_forcing: Ground truth for teacher forcing (batch, pred_len, N, C)
            tf_ratio: Teacher forcing ratio
        Returns:
            outputs: (batch, pred_len, N, out_channels)
        """
        batch, N, _ = h_list[0].shape
        device = h_list[0].device

        # Start with zero input
        decoder_input = torch.zeros(batch, N, self.out_channels, device=device)
        outputs = []

        for t in range(pred_len):
            for layer_idx, cell in enumerate(self.cells):
                inp = decoder_input if layer_idx == 0 else h_list[layer_idx - 1]
                h_list[layer_idx] = cell(inp, h_list[layer_idx], supports)

            output = self.projection(h_list[-1])  # (batch, N, out_channels)
            outputs.append(output)

            # Teacher forcing
            if teacher_forcing is not None and torch.rand(1).item() < tf_ratio:
                decoder_input = teacher_forcing[:, t]
            else:
                decoder_input = output

        return torch.stack(outputs, dim=1)  # (batch, pred_len, N, C)


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network.
    Encoder-Decoder architecture with diffusion graph convolution.
    """

    def __init__(self, num_sensors, num_supports=4, seq_len=12, pred_len=12,
                 hidden_dim=64, num_layers=2):
        """
        Args:
            num_sensors: Number of sensor nodes.
            num_supports: Number of diffusion support matrices.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            hidden_dim: Hidden state dimension.
            num_layers: Number of DCGRU layers.
        """
        super().__init__()
        self.num_sensors = num_sensors
        self.pred_len = pred_len

        self.encoder = DCRNNEncoder(num_supports, 1, hidden_dim, num_layers)
        self.decoder = DCRNNDecoder(num_supports, hidden_dim, 1, num_layers)

    def forward(self, x, supports, targets=None, tf_ratio=0.0):
        """
        Args:
            x: (batch, seq_len, num_sensors)
            supports: list of diffusion matrices (tensors on device)
            targets: (batch, pred_len, num_sensors) for teacher forcing
            tf_ratio: Teacher forcing ratio (decay during training)
        Returns:
            (batch, pred_len, num_sensors)
        """
        # Add feature dim: (batch, seq_len, N) → (batch, seq_len, N, 1)
        x = x.unsqueeze(-1)

        # Encode
        h_list = self.encoder(x, supports)

        # Prepare teacher forcing targets
        tf_targets = None
        if targets is not None and tf_ratio > 0:
            tf_targets = targets.unsqueeze(-1)

        # Decode
        output = self.decoder(h_list, supports, self.pred_len,
                              tf_targets, tf_ratio)

        return output.squeeze(-1)  # (batch, pred_len, N)

    def get_name(self):
        return "DCRNN"
