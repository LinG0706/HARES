from pathlib import Path
import sys

import pandas as pd
import torch
from torch_geometric.loader import DataLoader


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_sgformer_rgcn_hmoe_protocol import (  # noqa: E402
    ManifestGraphDataset,
    SGFormerRGCNQoRRegressor,
    design_feature_dim,
)


def main():
    root = HERE.parents[2]
    manifest = pd.read_csv(root / "hares_release/artifacts/protocol/manifest_kmeans_hmoe_source_exact_seed1.csv")
    target = manifest[manifest["dataset_role"] == "target"]
    cache_root = next((root / "HierarchicalMoE-master/save/programl").glob("v21_MLP-True*"))
    dataset = ManifestGraphDataset(target.iloc[:2], cache_root)
    torch.manual_seed(7)
    model = SGFormerRGCNQoRRegressor(
        design_feature_dim(dataset[0]), 16, 2, 1, 1, 3, 21, 0.0, "shared"
    ).eval()
    with torch.no_grad():
        alone = torch.cat([model(batch) for batch in DataLoader(dataset, batch_size=1)], dim=0)
        together = model(next(iter(DataLoader(dataset, batch_size=2))))
    torch.testing.assert_close(alone, together, atol=1e-4, rtol=1e-5)
    print("batch-independent predictions verified")


if __name__ == "__main__":
    main()
