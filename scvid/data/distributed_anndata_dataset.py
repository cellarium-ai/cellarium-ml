from scipy.sparse import issparse
from torch import Tensor
from torch.utils.data import Dataset

from . import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    def __init__(self, dadc: DistributedAnnDataCollection) -> None:
        self.dadc = dadc

    def __len__(self) -> int:
        return len(self.dadc)

    def __getitem__(self, idx: int) -> Tensor:
        """Return gene counts for a cell at idx."""
        X = self.dadc[idx].X

        return X.toarray() if issparse(X) else X
