import typing as t

from torch import Tensor
from torch.utils.data import Dataset

from . import DistributedAnnDataCollection


class DistributedAnnDataCollectionDataset(Dataset):
    def __init__(self, dac: DistributedAnnDataCollection) -> None:
        self.dac = dac
        self.dac.convert = convert
        outputs = []
        for name, value in convert.items():
            if isinstance(value, dict):
                for key in value.items():
                    outputs.append(dac.lazy_attr(name, key)
            else:
                outputs.append(dac.lazy_attr(name))
         self.outputs = outputs   # need to order values in the list or have it as a dict

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
