# %% [markdown]
# #### Customized Whisper Finetuning With Lora Implementation
# 
# In this notebook, we will implement the Lora finetuning 

# %%
import datasets
from datasets import load_dataset, DatasetDict,  Audio, load_from_disk
import pandas as pd
import os
import numpy as np
from tqdm import tqdm, tqdm_notebook
import datetime
import matplotlib.pyplot as plt
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import WhisperFeatureExtractor, AdamW
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, AutoTokenizer, AutoModelForCTC, Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score, classification_report, accuracy_score, roc_auc_score, roc_curve

# %%
# Load the whisper model
whisper_model = whisper.load_model('small')

# Get the encoder from the whisper model
encoder = whisper_model.encoder

# Set the device to CUDA if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class two_channel_ligo_binary_classifier(nn.Module):
    """
    A binary classifier model for two-channel LIGO data.

    Args:
        encoder (nn.Module): The encoder module for feature extraction.
        num_classes (int, optional): The number of output classes. Defaults to 1.

    Attributes:
        encoder (nn.Module): The encoder module for feature extraction.
        classifier (nn.Sequential): The classifier module for classification.

    Methods:
        forward(mel_tensor_0, mel_tensor_1): Performs forward pass of the model.

    """

    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.ln_post.normalized_shape[0] * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_tensor_0, mel_tensor_1):
        """
        Performs forward pass of the model.

        Args:
            mel_tensor_0 (torch.Tensor): Input mel spectrogram tensor for channel 0.
            mel_tensor_1 (torch.Tensor): Input mel spectrogram tensor for channel 1.

        Returns:
            torch.Tensor: Output logits of the model.

        """
        output_h1 = self.encoder(mel_tensor_0)[:, -1, :]
        output_l1 = self.encoder(mel_tensor_1)[:, -1, :]
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        return logits

# %%
model_ft = two_channel_ligo_binary_classifier(encoder)

# unfreeze all parameters for full finetuning
for param in model_ft.parameters():
    param.requires_grad = True

# count the trainable of parameters
num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
# print parameters in millions
print(f"Number of parameters: {num_params/1e6}M")

# %%
print(model_ft)

# %%
model_freeze_encoder = two_channel_ligo_binary_classifier(encoder)

# freeze the encoder parameters
for param in model_freeze_encoder.encoder.parameters():
    param.requires_grad = False

# count the trainable of parameters
num_params = sum(p.numel() for p in model_freeze_encoder.parameters() if p.requires_grad)
# print parameters in millions
print(f"Number of trainable parameters: {num_params/1e6}M")

# %% [markdown]
# #### Coding Lora from scratch

# %%
class LoRA_layer(torch.nn.Module):
    """
    LoRA_layer is a custom PyTorch module that implements the LoRA (Low-Rank Adaptation) mechanism.

    Args:
        in_dim (int): The input dimension of the layer.
        out_dim (int): The output dimension of the layer.
        rank (int): The rank of the low-rank approximation.
        alpha (float): The scaling factor for the output.

    Attributes:
        A (torch.nn.Parameter): The learnable parameter representing the low-rank matrix A.
        B (torch.nn.Parameter): The learnable parameter representing the low-rank matrix B.
        alpha (float): The scaling factor for the output.
        rank (int): The rank of the low-rank approximation.

    Methods:
        forward(x): Performs the forward pass of the LoRA layer.

    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = self.alpha / self.rank * (x @ self.A @ self.B)
        return x

class LoRa_linear(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRA_layer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# %%
whisper_lora = two_channel_ligo_binary_classifier(encoder)
# freeze the parameters of the model
for param in whisper_lora.encoder.parameters():
    param.requires_grad = False

# %%
class lora_params:
    rank = 4
    alpha = 4
    query = True
    key = True
    mlp = True
    value = True
    head = False

# replace the linear layer with LoRA_linear
replace_lora = partial(LoRa_linear, rank=lora_params.rank, alpha=lora_params.alpha)

for layer in whisper_lora.encoder.blocks:
    if lora_params.query:
        layer.attn.query = replace_lora(layer.attn.query)
    if lora_params.key:
        layer.attn.key = replace_lora(layer.attn.key)
    if lora_params.value:
        layer.attn.value = replace_lora(layer.attn.value)
    if lora_params.mlp:
        for i, mlp_layer in enumerate(layer.mlp):
            if isinstance(mlp_layer, torch.nn.Linear):
                layer.mlp[i] = replace_lora(mlp_layer)
        
if lora_params.head:
    for layer in whisper_lora.classifier:
        layer = replace_lora(layer)

print(whisper_lora)

# %% [markdown]
# Now all the Q K V layer in the multihead attention has been replaced by the Lora layer.
# This implementation can also be applied to any other other model with linear layers.

# %%
# count the trainable of parameters
num_params = sum(p.numel() for p in whisper_lora.parameters() if p.requires_grad)
# print parameters in millions
print(f"Number of trainable parameters: {num_params/1e6}M")

# %% [markdown]
# And now we have three model to compare, the original **model with full parameters fine-tuning**, the original **model with encoder part freezed**, and the original **model with the encoder part freezed and applied lora linear layer**. The trainable parameters of the three models are **89M**, **2.32M** and **2.82M** respectively. We will compare the performance of the three models.

# %%



