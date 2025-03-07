import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import time



class RawAudioDataset(Dataset):
  
    def __init__(self,
                 audio_files,
                 sample_rate=16000,
                 raw_context_samples=40960,
                 raw_future_samples=20480,    
                 stride=16000):    
        super().__init__()
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.raw_context_samples = raw_context_samples
        self.raw_future_samples = raw_future_samples
        self.stride = stride

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx]
        waveform, sr = torchaudio.load(path)  # => [channels, total_samples]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        total_samples = waveform.size(1)
        needed = self.raw_context_samples + self.raw_future_samples
        if total_samples < needed:
            return None  # skip or you could pad

        chunks = []
        start = 0
        while start + needed <= total_samples:
            # context chunk:  [start : start + raw_context_samples]
            # future chunk:   [start + raw_context_samples : start + needed]
            ctx = waveform[:, start : start + self.raw_context_samples]
            fut = waveform[:, start + self.raw_context_samples : start + needed]
            chunks.append((ctx, fut))
            start += self.stride
        if len(chunks) == 0:
            return None
        return chunks

def raw_collate_fn(batch):
    """
    'batch' is a list of 'chunks-lists', one per file. 
    We flatten them into a single list of (context, future).
    Then stack => [N, 1, raw_samples].
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    contexts = []
    futures = []
    for chunk_list in batch:
        for (c, f) in chunk_list:
            contexts.append(c)
            futures.append(f)

    contexts = torch.stack(contexts, dim=0)  # => [N, 1, raw_context_samples]
    futures = torch.stack(futures, dim=0)   # => [N, 1, raw_future_samples]
    return contexts, futures



class LibriSpeechDataset(Dataset):
    def __init__(self, audio_files, waveform_length, context_length, future_length, negative_waveform_length):
        self.audio_files = audio_files
        self.waveform_length = waveform_length
        self.context_length = context_length
        self.future_length = future_length
        self.negative_waveform_length = negative_waveform_length

    def __len__(self):
        return len(self.audio_files)

    def load_waveform(self, audio_path, waveform_length):
        waveform, _ = torchaudio.load(audio_path)
        if waveform.size(1) > waveform_length:
            start_idx = random.randint(0, waveform.size(1) - waveform_length)
            waveform = waveform[:, start_idx: start_idx + waveform_length]
        else:
            pad_length = waveform_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.load_waveform(audio_path, self.waveform_length)

        # Generate context waves
        start_idx = random.randint(0, self.waveform_length - self.context_length - self.future_length)
        context = waveform[:, start_idx: start_idx + self.context_length]

        # Generate future samples
        future = waveform[:, start_idx + self.context_length: start_idx + self.context_length + self.future_length]

        # Generate negative sample
        negative_idx = random.randint(0, len(self.audio_files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.audio_files) - 1)

        negative_audio_path = self.audio_files[negative_idx]
        negative_waveform = self.load_waveform(negative_audio_path, self.negative_waveform_length)

        negative_sample = negative_waveform

        # Return context, future, negative sample, and waveform length
        return context, future, negative_sample, context.size(1)
