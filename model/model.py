import os
# from ctcdecode import CTCBeamDecoder
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from wer import calculate_wer
# import torchmetrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from conformer import Conformer
import time


# class ConvFeatureExtractor(nn.Module):
  
#     def __init__(self):
#         super().__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=256, kernel_size=20, stride=10, padding=5),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, stride=16, padding=8),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.conv_net(x)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, z_size=512):
        super(ConvFeatureExtractor, self).__init__()
        self.z_size = z_size
        
        # Reuse the same CNN architecture from the original code
        # with a downsampling factor of 160
        self.feature_conv = nn.Sequential( 
            nn.Conv1d(1, self.z_size, kernel_size=10, stride=5, padding=3, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: (batch_size, channels=1, seq_len)
        output: (batch_size, z_size, seq_len // 160)
        """
        return self.feature_conv(x)

class Conformer_JUST(nn.Module):

    def __init__(self, encoder_dim=512, num_encoder_layers=6, num_classes = 1000):
        super().__init__()
        self.conv_feature_extractor = ConvFeatureExtractor()

        self.encoder = Conformer(
            num_classes=encoder_dim,
            input_dim=encoder_dim,  # 512
            encoder_dim=encoder_dim,
            num_attention_heads=8,
            num_encoder_layers=num_encoder_layers
        )

        # A simple multi-step predictor
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
        )

        self.predictor_ctc = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, num_classes)
        )

    def encode_frames(self, x):
        """
        x: [B, 1, raw_samples]
        Returns encoded features and computed lengths.
        """
        # Pass raw audio through conv front-end:
        feats = self.conv_feature_extractor(x)  # => [B, 512, T_out]
        
        # Compute T_out from the conv output shape:
        T_out = feats.size(2)
        # Create a lengths tensor (assuming all samples have the same length here):
        lengths = torch.full((x.size(0),), T_out, dtype=torch.int64, device=x.device)
        
        # Transpose to match Conformer expected shape: [B, T_out, 512]
        feats = feats.transpose(1, 2)
        
        # Pass to encoder along with lengths:
        encoded, lens = self.encoder(feats, lengths)
        return encoded

    
    def encoder_ctc(self, x, lengths):

        x, out_length = self.encoder(x, lengths)
        lan_out = self.predictor_ctc(x)
        lan_out = F.log_softmax(lan_out, dim=-1)

        return lan_out, out_length


    def forward(self, context_wave = None, future_wave = None, mel_feat = None, mel_lengths = None):
        """
        context_wave: [B, 1, raw_context_samples]
        future_wave:  [B, 1, raw_future_samples]
        returns predicted_future, true_future => [B, T_f, 512]
        where T_f ~ raw_future_samples / 160
        """
        if (context_wave) and (future_wave is not None):

            context_encoded = self.encode_frames(context_wave)  # => [B, T_c, 512]
            future_encoded  = self.encode_frames(future_wave)   # => [B, T_f, 512]

            last_context = context_encoded[:, -1, :]   # [B, 512]
            T_f = future_encoded.size(1)

            # multi-step predictor
            preds = []
            current_vec = last_context
            for _ in range(T_f):
                p = self.predictor(current_vec)  # => [B, 512]
                preds.append(p)
                # optionally chain it:
                current_vec = p
            predicted_future = torch.stack(preds, dim=1)  # => [B, T_f, 512]

            return predicted_future, future_encoded
        
        if mel_feat is not None:
            lan_out, out_lengths = self.encoder_ctc(mel_feat, mel_lengths)
            
            return lan_out, out_lengths
