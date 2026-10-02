# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from conftest import make_counts, same

torch = pytest.importorskip("torch")

from deltacells import DatasetWriter, open_dataset  # noqa: E402

N, G, TILE = 240, 50, 60


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("t") / "ds")
    full = make_counts(N, G, density=0.2, seed=50)
    with DatasetWriter(root, n_genes=G, tile_size=TILE, n_chunks=2, level=3) as w:
        for lo in range(0, N, TILE):
            w.add_tile(full[lo : lo + TILE])
    return root, full


def test_to_torch_csr_matches_dense(built):
    root, full = built
    b = open_dataset(root).get_batch([7, 3, 200, 3])
    t = b.to_torch_csr()
    assert t.layout == torch.sparse_csr and t.shape == (4, G) and t.dtype == torch.float32
    assert torch.equal(t.to_dense(), torch.from_numpy(full[[7, 3, 200, 3]].toarray()))
    assert np.shares_memory(t.col_indices().numpy(), b.indices)  # no copy of the big arrays


def test_to_torch_csr_empty_batch(built):
    t = open_dataset(built[0]).get_batch([]).to_torch_csr()
    assert t.shape == (0, G)


def test_shared_memory_output_buffers(built):
    root, full = built
    ds = open_dataset(root)
    idx = np.arange(10, 90)
    need = full[idx].nnz
    oi, ov = (
        torch.empty(need, dtype=torch.int32).share_memory_(),
        torch.empty(need, dtype=torch.float32).share_memory_(),
    )
    b = ds.get_batch(idx, out_indices=oi.numpy(), out_values=ov.numpy())
    assert same(b.to_scipy(), full[idx]) and oi.is_shared()


class _Rows(torch.utils.data.IterableDataset):
    """Each worker yields the batches ``range(i*B, (i+1)*B)`` for its share of the batch numbers."""

    def __init__(self, ds, batch_size):
        self.ds, self.batch_size = ds, batch_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        n_batches = -(-self.ds.n_cells // self.batch_size)
        for b in range(wid, n_batches, nw):
            idx = np.arange(b * self.batch_size, min((b + 1) * self.batch_size, self.ds.n_cells))
            self.ds.prefetch_cells(idx)
            batch = self.ds.get_batch(idx)
            yield {
                "id": b,
                "indptr": torch.from_numpy(batch.indptr.astype(np.int32)),
                "indices": torch.from_numpy(batch.indices),
                "values": torch.from_numpy(batch.values),
            }


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_with_workers(built, workers):
    root, full = built
    ds = open_dataset(root, max_cached_tiles=1)
    dl = torch.utils.data.DataLoader(_Rows(ds, 25), batch_size=None, num_workers=workers)
    got = {}
    for item in dl:
        csr = torch.sparse_csr_tensor(
            item["indptr"], item["indices"], item["values"], size=(len(item["indptr"]) - 1, G)
        )
        got[int(item["id"])] = csr.to_dense().numpy()
    assert sorted(got) == list(range(10))
    assert np.array_equal(np.vstack([got[b] for b in sorted(got)]), full.toarray())
