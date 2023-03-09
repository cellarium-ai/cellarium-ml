from typing import Dict, Iterable, Tuple

import torch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"X": self.data[idx]}


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.iter_data = []
        self.idx = torch.nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _get_fn_args_from_batch(
        tensor_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Iterable, dict]:
        return (), tensor_dict

    def forward(self, **batch):
        self.iter_data.append(batch)
        loss = batch["X"].sum() * self.idx
        return loss
