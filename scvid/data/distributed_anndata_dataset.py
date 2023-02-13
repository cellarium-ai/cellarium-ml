from scipy.sparse import issparse
from torch import Tensor
from torch.utils.data import Dataset

from . import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    def __init__(self, dadc: DistributedAnnDataCollection) -> None:
        self.dadc = dadc
        convert = {"X": lambda a: a.toarray() if issparse(a) else a}  # densify .X
        self.dadc.convert = convert

    def __len__(self) -> int:
        return len(self.dadc)

    def __getitem__(self, idx: int) -> Tensor:
        """Return gene counts for a cell at idx."""

        return self.dadc[idx].X
