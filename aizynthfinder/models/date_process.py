from torch.utils.data import Dataset


class BaseData(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.target = label

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


class ExpansionData(BaseData):
    pass


class FilterData(BaseData):
    pass
