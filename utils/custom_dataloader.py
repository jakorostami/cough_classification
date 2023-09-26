import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")


class CoughData(Dataset):
    
    def __init__(self, mel_path, annotations_file):
        self.mel_path = mel_path
        self.annotations = pd.read_csv(annotations_file)
        
    
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        mel_sample_path = self._get_mel_path(idx)
        label = self._get_mel_sample_label(idx)
        
        mel = torch.tensor(torch.load(mel_sample_path)).unsqueeze_(0)   # Accidentally saved topological data as numpy arrays therefore this is needed
        
        return mel, label
        
    
    def _get_mel_path(self, idx):
        
        path = os.path.join(self.mel_path, self.annotations.iloc[idx, 2])
        return path
    
    def _get_mel_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]
    
    
