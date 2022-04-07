import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from einops import repeat, rearrange, reduce


class LSTMStepwiseFusion(nn.Module):
    def __init__(
        self,
        init_signal_size,  # input for initial lstm states
        driven_signal_size,  # input other step-wise feature, e.g. speaker_av
        lstm_input_size,
        act_layer,
        hidden_size,
        num_layers=1,
        bias=False,
        batch_first=False,
        dropout=0,
    ):
        super().__init__()
        self.layernorm_init = nn.LayerNorm(init_signal_size)
        self.layernorm_driven = nn.LayerNorm(driven_signal_size)
        self.get_initial_state = nn.Sequential(
            nn.Linear(init_signal_size, init_signal_size),
            act_layer(),
            nn.Linear(init_signal_size, 2 * num_layers * hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(driven_signal_size, lstm_input_size),
            act_layer(),
        )
        self.lstm_layers = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional=False,
        )
        self.lstm_num_layers = num_layers
        self.lstm_hidden_size = hidden_size

    def forward(
        self, 
        driven,
        init,
        lengths,
    ):
        init = self.get_initial_state(self.layernorm_init(init))
        init_hidden, init_cell = init.view(-1, 2, self.lstm_num_layers, self.lstm_hidden_size).permute(1, 2, 0, 3).contiguous()

        driven = self.layernorm_driven(driven)
        fused_features = self.fusion(driven)
        lengths = lengths.cpu().tolist()
        fused_features = rnn_utils.pack_padded_sequence(fused_features, lengths, batch_first=True, enforce_sorted=False)

        fused_features, _ = self.lstm_layers(fused_features, (init_hidden, init_cell))

        fused_features_unpacked, length_unpacked = rnn_utils.pad_packed_sequence(fused_features, batch_first=True)

        return fused_features_unpacked
