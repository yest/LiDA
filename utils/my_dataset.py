from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = torch.tensor(self.label[idx])
        return {'text': text, 'label':label}