from pathlib import Path
import argparse
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, global_add_pool

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
NEW4_SCRIPT_DIR = EXPERIMENTS_DIR / "new4+harp" / "scripts"
for _path in (EXPERIMENTS_DIR, NEW4_SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from baseline_hmoe_common import (  # noqa: E402
    add_common_args, count_parameters, finish_run, train_source,
    train_support_finetune,
)
from train_old_expert import (  # noqa: E402
    TARGETS, as_scalar_tensor, design_feature_dim, design_node_features,
    infer_cache_root, select_rows, set_seed,
)


def edge_types_from_cache(data, num_relations):
    edge_type = getattr(data, "edge_type", None)
    if edge_type is not None:
        return edge_type.long().clamp_min(0).clamp_max(num_relations - 1)
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros(data.edge_index.size(1), dtype=torch.long, device=data.edge_index.device)
    raw = edge_attr[:, 0].round().long()
    edge_type = torch.zeros_like(raw)
    edge_type[raw == 1] = 1
    if num_relations > 2:
        edge_type[raw == 4] = 2
    return edge_type.clamp_max(num_relations - 1)


def linear_attention(qs, ks, vs, graph_ids, num_graphs, chunk_size=512):
    """Apply relation attention independently to every graph in a batch."""
    n_items, num_heads, hidden_dim = qs.shape
    kv_sums = qs.new_zeros((num_graphs, num_heads, hidden_dim, hidden_dim))
    k_sums = qs.new_zeros((num_graphs, num_heads, hidden_dim))
    counts = qs.new_zeros((num_graphs, 1, 1))

    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        graph_chunk = graph_ids[start:stop]
        k_chunk, v_chunk = ks[start:stop], vs[start:stop]
        kv_sums.index_add_(0, graph_chunk, k_chunk.unsqueeze(-1) * v_chunk.unsqueeze(-2))
        k_sums.index_add_(0, graph_chunk, k_chunk)
        counts.index_add_(0, graph_chunk, qs.new_ones((stop - start, 1, 1)))

    outputs = []
    for start in range(0, n_items, chunk_size):
        stop = min(start + chunk_size, n_items)
        graph_chunk = graph_ids[start:stop]
        q_chunk, v_chunk = qs[start:stop], vs[start:stop]
        kv_chunk = kv_sums.index_select(0, graph_chunk)
        k_chunk_sum = k_sums.index_select(0, graph_chunk)
        n_chunk = counts.index_select(0, graph_chunk)
        numerator = torch.einsum("ehm,ehmd->ehd", q_chunk, kv_chunk) + n_chunk * v_chunk
        denominator = torch.einsum("ehm,ehm->eh", q_chunk, k_chunk_sum).unsqueeze(-1) + n_chunk
        outputs.append(numerator / denominator.clamp_min(1e-6))
    return torch.cat(outputs, dim=0)


class RelationGlobalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_relations):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False) for _ in range(num_relations)])
        self.query = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False) for _ in range(num_relations)])
        self.value = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False) for _ in range(num_relations)])
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, edge_index, edge_type, batch):
        row, col = edge_index
        output = x.new_zeros((x.size(0), self.hidden_dim))
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        for rel in torch.unique(edge_type, sorted=True).tolist():
            mask = edge_type == int(rel)
            rows, cols = row[mask], col[mask]
            if rows.numel() == 0:
                continue
            qs = self.query[rel](x[rows]).view(-1, self.num_heads, self.hidden_dim)
            ks = self.key[rel](x[cols]).view(-1, self.num_heads, self.hidden_dim)
            vs = self.value[rel](x[cols]).view(-1, self.num_heads, self.hidden_dim)
            attended = linear_attention(qs, ks, vs, batch[rows], num_graphs)
            output.index_add_(0, rows, attended.mean(dim=1))
        return self.output(output)


class SGFormerBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers, heads, relations, dropout):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([RelationGlobalAttention(hidden_dim, heads, relations) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = dropout
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_type, batch):
        x = F.dropout(F.relu(self.input(x)), p=self.dropout, training=self.training)
        for layer, norm in zip(self.layers, self.norms):
            update = layer(x, edge_index, edge_type, batch)
            w = self.residual_weight.sigmoid()
            x = F.relu(norm(w * update + (1.0 - w) * x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SGFormerRGCNBackbone(nn.Module):
    def __init__(self, in_dim, hidden_dim, rgcn_layers, sgformer_layers, heads, relations, global_dim, dropout):
        super().__init__()
        self.relations = relations
        self.global_dim = global_dim
        self.dropout = dropout
        self.sgformer = SGFormerBranch(in_dim, hidden_dim, sgformer_layers, heads, relations, dropout)
        self.rgcn = nn.ModuleList([
            RGCNConv(in_dim if i == 0 else hidden_dim, hidden_dim, num_relations=relations)
            for i in range(rgcn_layers)
        ])
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.out_dim = hidden_dim + global_dim

    def global_features(self, data, n_graphs):
        values = getattr(data, "pragmas", None)
        if values is None:
            return data.x.new_zeros((n_graphs, self.global_dim))
        values = values.to(device=data.x.device, dtype=data.x.dtype).view(n_graphs, -1)
        if values.size(1) < self.global_dim:
            values = F.pad(values, (0, self.global_dim - values.size(1)))
        return values[:, :self.global_dim]

    def forward(self, data):
        x = design_node_features(data)
        et = edge_types_from_cache(data, self.relations)
        gx = self.sgformer(x, data.edge_index, et, data.batch)
        lx = x
        for conv in self.rgcn:
            lx = F.dropout(F.relu(conv(lx, data.edge_index, et)), p=self.dropout, training=self.training)
        w = self.fusion_weight.sigmoid()
        graph = global_add_pool(w * lx + (1.0 - w) * gx, data.batch)
        return torch.cat([graph, self.global_features(data, graph.size(0))], dim=1)


def make_head(dim, dropout):
    return nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))


class SGFormerRGCNQoRRegressor(nn.Module):
    accepts_data_batch = True

    def __init__(self, in_dim, hidden_dim, rgcn_layers, sgformer_layers, heads, relations, global_dim, dropout, mode):
        super().__init__()
        self.predictor_mode = mode
        if mode == "shared":
            self.backbone = SGFormerRGCNBackbone(in_dim, hidden_dim, rgcn_layers, sgformer_layers, heads, relations, global_dim, dropout)
            self.heads = nn.ModuleDict({target: make_head(self.backbone.out_dim, dropout) for target in TARGETS})
        else:
            self.models = nn.ModuleDict()
            for target in TARGETS:
                backbone = SGFormerRGCNBackbone(in_dim, hidden_dim, rgcn_layers, sgformer_layers, heads, relations, global_dim, dropout)
                self.models[target] = nn.ModuleDict({"backbone": backbone, "head": make_head(backbone.out_dim, dropout)})

    def forward(self, data):
        if self.predictor_mode == "shared":
            hidden = self.backbone(data)
            return torch.cat([self.heads[target](hidden) for target in TARGETS], dim=1)
        return torch.cat([self.models[target]["head"](self.models[target]["backbone"](data)) for target in TARGETS], dim=1)


class ManifestGraphDataset(torch.utils.data.Dataset):
    def __init__(self, frame, cache_root):
        self.rows = frame.reset_index(drop=True).to_dict(orient="records")
        self.cache_root = Path(cache_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = torch.load(self.cache_root / str(row["cache_split"]) / str(row["file_name"]), map_location="cpu", weights_only=False)
        edge_attr = getattr(raw, "edge_attr", None)
        raw_edge_type = edge_attr[:, 0].round().long() if edge_attr is not None and edge_attr.numel() else torch.zeros(raw.edge_index.size(1), dtype=torch.long)
        data = Data(x=raw.x.contiguous(), edge_index=raw.edge_index.contiguous(), edge_type=raw_edge_type.contiguous(), X_pragma_per_node=getattr(raw, "X_pragma_per_node", None), X_pragmascopenids=getattr(raw, "X_pragmascopenids", None), pragmas=getattr(raw, "pragmas", torch.zeros(1, 21)).contiguous())
        for name in ("kernel", "gname", "key", "sample_id", "dataset_role", "cache_split", "file_name"):
            setattr(data, name, str(row[name]))
        data.manifest_split = str(row["split"])
        data.support_query = "" if pd.isna(row.get("support_query", "")) else str(row.get("support_query", ""))
        data.actual_perf = as_scalar_tensor(getattr(raw, "actual_perf", row["actual_perf"]))
        for target in TARGETS:
            setattr(data, target, as_scalar_tensor(getattr(raw, target, row[target])))
        return data


def load_protocol_inputs_lazy(args):
    set_seed(args.seed)
    manifest_path = Path(args.manifest)
    cache_root = infer_cache_root(manifest_path, args.cache_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    if args.phase == "source":
        frames = {"train": select_rows(manifest, "source", split="train"), "source_val": select_rows(manifest, "source", split="val"), "source_test": select_rows(manifest, "source", split="test"), "target_support": select_rows(manifest, "target", split="support", k=args.k), "target_query": select_rows(manifest, "target", split="query", k=args.k)}
        if args.max_source_train > 0:
            frames["train"] = frames["train"].sample(n=min(args.max_source_train, len(frames["train"])), random_state=args.seed)
        if args.max_eval > 0:
            for name in ("source_val", "source_test", "target_support", "target_query"):
                frames[name] = frames[name].sample(n=min(args.max_eval, len(frames[name])), random_state=args.seed)
    else:
        frames = {"train": select_rows(manifest, "target", split="support", k=args.k), "target_support": select_rows(manifest, "target", split="support", k=args.k), "target_query": select_rows(manifest, "target", split="query", k=args.k)}
        if args.max_support > 0:
            frames["train"] = frames["train"].sample(n=min(args.max_support, len(frames["train"])), random_state=args.seed)
            frames["target_support"] = frames["target_support"].sample(n=min(args.max_support, len(frames["target_support"])), random_state=args.seed)
        if args.max_query > 0:
            frames["target_query"] = frames["target_query"].sample(n=min(args.max_query, len(frames["target_query"])), random_state=args.seed)
    loaders = {name: ManifestGraphDataset(frame, cache_root) for name, frame in frames.items()}
    return manifest_path, cache_root, out_dir, loaders, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_model="sgformer_rgcn_qor")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--rgcn_layers", type=int, default=2)
    parser.add_argument("--sgformer_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_relations", type=int, default=3)
    parser.add_argument("--global_dim", type=int, default=21)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predictor-mode", choices=["shared", "independent"], default="shared")
    args = parser.parse_args()
    manifest_path, cache_root, out_dir, loaders, device = load_protocol_inputs_lazy(args)
    in_dim = design_feature_dim(loaders["train"][0])
    model = SGFormerRGCNQoRRegressor(in_dim, args.hidden_dim, args.rgcn_layers, args.sgformer_layers, args.num_heads, args.num_relations, args.global_dim, args.dropout, args.predictor_mode).to(device)
    print("Protocol: HMOE-aligned SGFormer-RGCN adapter; mode=", args.predictor_mode, "params=", count_parameters(model), "rows=", {k: len(v) for k, v in loaders.items()})
    if args.phase == "source":
        train_info = train_source(args, model, loaders, device, out_dir, f"sgformer_source seed={args.seed} k={args.k}")
    else:
        train_info = train_support_finetune(args, model, loaders, device, out_dir, f"sgformer_support_ft seed={args.seed} k={args.k}")
    finish_run(args=args, model=model, loaders=loaders, device=device, out_dir=out_dir, manifest_path=manifest_path, cache_root=cache_root, baseline_family="SGFormer-RGCN", baseline_component="SGFormer global attention + RGCN local relation QoR predictor", architecture={"input_dim": in_dim, "hidden_dim": args.hidden_dim, "rgcn_layers": args.rgcn_layers, "sgformer_layers": args.sgformer_layers, "num_heads": args.num_heads, "num_relations": args.num_relations, "global_dim": args.global_dim, "predictor_mode": args.predictor_mode, "protocol_note": "HMoE cache pragma features replace unavailable original HLS-report GF"}, train_info=train_info)


if __name__ == "__main__":
    main()
