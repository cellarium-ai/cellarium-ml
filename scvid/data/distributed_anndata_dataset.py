import typing as t

from torch import Tensor
from torch.utils.data import Dataset

from . import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    def __init__(self, dac: DistributedAnnDataCollection) -> None:
        self.dac = dac

    def __len__(self) -> int:
        return len(self.dac)

    def __getitem__(self, index: int) -> t.Tuple[Tensor, Tensor]:
        """
        :return: Tuple of tensor with a cell gene counts and db index.
        """

        v = self.dac[index]
        x_i = Tensor(v.X.todense().astype(int))

        # obs_names is now "cell_<int>" which we are splitting out
        # to return the db_index tensor
        db_index = Tensor(v.obs_names.str.split("_").str[-1].astype(int))

        return x_i, db_index
