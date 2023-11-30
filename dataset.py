import torch
from torch.utils.data import Dataset




class mydateset(Dataset):
    def __init__(self):
        super(mydateset, self).__init__()

        pass


    def __len__(self):
        return len(self.train_list)
    

    def __getitem__(self, index):
        return self.train_list[index]