import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

class LigoBinaryData(Dataset):
    def __init__(self, ds, device):
        self.ds = ds
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        audio, label = self.ds[idx]['timeseries'], self.ds[idx]['labels']
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        label = torch.tensor([1, 0]) if label == 0 else torch.tensor([0, 1])
        input_features = inputs.input_features.squeeze(0)
        return input_features, label

class two_channel_LigoBinaryData(Dataset):
    def __init__(self, ds, device):
        self.ds = ds
        self.device = device
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        h1_audio, l1_audio, label, snr = self.ds[idx]['h1_timeseries'], self.ds[idx]['l1_timeseries'], self.ds[idx]['labels'], self.ds[idx]['injection_snr']
        
        h1_inputs = self.feature_extractor(h1_audio, sampling_rate=16000, return_tensors="pt")
        l1_inputs = self.feature_extractor(l1_audio, sampling_rate=16000, return_tensors="pt")
        
        h1_input_features = h1_inputs.input_features.squeeze(0)
        l1_input_features = l1_inputs.input_features.squeeze(0)

        return h1_input_features, l1_input_features, label, snr