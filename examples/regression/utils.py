import torch
import torch.utils.data as data

class SynthDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def build_model(input_dim, output_dim=1, initial_value=None):
    model = torch.nn.Sequential()
    module = torch.nn.Linear(input_dim, output_dim, bias=False)
    if initial_value is not None:
        module.weight.data = torch.from_numpy(initial_value).type(torch.FloatTensor)
        model.add_module("linear", module)
    else:
        model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))
    return model