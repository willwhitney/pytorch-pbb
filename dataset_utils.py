import torch
from torch.utils.data import Dataset


class DatasetCache(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.cache = {}

    def __getitem__(self, index):
        with torch.no_grad():
            if index in self.cache: return self.cache[index]
            else:
                self.cache[index] = self.dataset[index]
                return self.cache[index]

    def __len__(self):
        return len(self.dataset)


class DatasetSubset(Dataset):
    def __init__(self, dataset, start=0, stop=None):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.stop = stop
        if stop is not None: 
            self.stop = min(self.stop, len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index + self.start]

    def __len__(self):
        stop = self.stop or len(self.dataset)
        return stop - self.start
