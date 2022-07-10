import torch
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data,args=None):
        self.data = data
        self.args = args
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):

        label = torch.tensor([item[0]['cls'] for item in data])
        tokens = [torch.tensor(item[0]['tokens']) for item in data]
        ind = torch.tensor([item[1] for item in data])
        return (
            label,
            tokens,
            ind
        )

def get_data_loader(args, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set(data, args)

    if batch_size == None:
        batch_size = min(args.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader





    