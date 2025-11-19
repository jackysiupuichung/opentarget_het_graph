#!/usr/bin/env python3
import argparse
import os
import datetime
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf


from src.data.utils import supervision_edge_temporal_and_cold_split, attach_node_features
from src.pipeline.build_progression_graph import load_nodes, load_edges, get_most_evidented_edges, build_heterodata_with_cold_split
from src.data.dataset import InteractionDataset
from src.models.base_lightning import NCFRecLightning, GraphRecLightning
from src.models.ncf import NCF

from src.models.utils import initialise_model, initialise_trainer

def build_all_interactions(df, user_map, item_map):
    all_interactions = {}
    for u, i in zip(df["user_id"], df["item_id"]):
        uid = user_map[str(u)]
        iid = item_map[str(i)]
        all_interactions.setdefault(uid, set()).add(iid)
    return all_interactions

def serialise_predictions(trainer, model, dataloader, run_dir, user_map, item_map, stage, is_graph=False):
    """
    Run predictions using trainer.predict, then save to CSV + save user/item maps.
    """
    # Run inference (list of batch outputs from predict_step)
    preds = trainer.predict(model, dataloaders=dataloader)

    # Flatten list of tensors
    if isinstance(preds[0], dict):
        # if predict_step returns dicts (more flexible)
        preds = {k: torch.cat([b[k] for b in preds]).cpu().numpy() for k in preds[0].keys()}
        users, items, labels, scores = preds["user"], preds["item"], preds["label"], preds["pred"]
    else:
        # if predict_step just returns scores
        scores = torch.cat(preds).cpu().numpy()
        # fall back: dataloader still holds user/item/labels
        users, items, labels = [], [], []
        for batch in dataloader:
            if is_graph:
                users.extend(batch.edge_label_index[0].cpu().tolist())
                items.extend(batch.edge_label_index[1].cpu().tolist())
                labels.extend(batch.edge_label.cpu().tolist())
            else:
                users.extend(batch["user_id"].cpu().tolist())
                items.extend(batch["item_id"].cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
        users, items, labels = map(torch.tensor, (users, items, labels))

    # Reverse maps
    rev_user_map = {v: k for k, v in user_map.items()}
    rev_item_map = {v: k for k, v in item_map.items()}

    user_names = [rev_user_map.get(int(u), u) for u in users]
    item_names = [rev_item_map.get(int(i), i) for i in items]

    # Save predictions CSV
    df = pd.DataFrame({
        "user_id": user_names,
        "item_id": item_names,
        "label": labels.tolist(),
        "pred": scores.tolist(),
    })
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    out_path = os.path.join(pred_dir, f"{stage}_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 {stage} predictions saved to {out_path}")

    # # Save user/item maps (JSON for reusability)
    # map_dir = os.path.join(run_dir, "mappings")
    # os.makedirs(map_dir, exist_ok=True)

    # with open(os.path.join(map_dir, "user_map.json"), "w") as f:
    #     json.dump(user_map, f)
    # with open(os.path.join(map_dir, "item_map.json"), "w") as f:
    #     json.dump(item_map, f)
    # print(f"💾 User/item maps saved to {map_dir}")

    return df


def main(cfg):
    pl.seed_everything(cfg.train.seed)
    # -----------------------
    # Step 0: Create run directory
    # -----------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{cfg.model.name}_{cfg.model.loss_type}_{cfg.data.cutoff}_{cfg.data.horizon}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save a copy of config into run_dir for reproducibility
    OmegaConf.save(config=cfg, f=os.path.join(run_dir, "config.yaml"))

    print(f"🚀 Starting run → {run_dir}")

    # -----------------------
    # Step 1: Custom temporal and user split based on pwas
    # -----------------------
    cold_start_diseases = []
    if cfg.data.cold_start_file and os.path.exists(cfg.data.cold_start_file):
        print(f"✅ Loaded cold start diseases from {cfg.data.cold_start_file}")
        cold_start_df = pd.read_csv(cfg.data.cold_start_file)
        cold_start_diseases = cold_start_df.iloc[:, 0].dropna().astype(str).tolist()
            

    train_df, valid_df, test_df = supervision_edge_temporal_and_cold_split(
        cfg.data.parquet,
        cutoff=cfg.data.cutoff,
        horizon=cfg.data.horizon,
        cold_start_diseases=cold_start_diseases,
        out_dir=run_dir
    )

    nodes, id_to_type = load_nodes(cfg.data.node_dir)
    edges = load_edges(cfg.data.edge_dir)
    # this include all evidence edges before cutoff
    edges = edges[edges['year'] <= cfg.data.cutoff]
    # TODO: based on datatype or datasource
    edges = get_most_evidented_edges(edges)

    # -----------------------
    # Step 3: Generate id_maps
    # -----------------------
    user_map = {nid: i for i, nid in enumerate(nodes["diseases"]["id"].astype(str).tolist())}
    item_map = {nid: i for i, nid in enumerate(nodes["targets"]["id"].astype(str).tolist())}
    print(f"✅ Built id_maps: {len(user_map)} diseases, {len(item_map)} targets")
    # This includes all interactions within the training set to avoid temporal leakage
    train_interactions = build_all_interactions(train_df, user_map, item_map)
    # -----------------------
    # Step 4: Build hetero graph
    # -----------------------
    hetero_graph = None
    if getattr(cfg.data, "graph_file", None) and os.path.exists(cfg.data.graph_file):
        print(f"✅ Loading precomputed hetero graph from {cfg.data.graph_file}")
        hetero_graph, id_maps = torch.load(cfg.data.graph_file, weights_only=False)

    else:
        print("⚙️ Building hetero graph from nodes/edges...")
        hetero_graph, id_maps = build_heterodata_with_cold_split(
            nodes,
            edges, 
            train_df, 
            valid_df, 
            test_df, 
            cfg.data.cutoff, 
            cfg.data.horizon,
            supervision_source=cfg.model.supervision_src_type, 
            supervision_target=cfg.model.supervision_dst_type, 
            supervision_relation=cfg.model.supervision_relation_type
        )
        # Save for reuse
        if getattr(cfg.data, "graph_file", None):
            os.makedirs(os.path.dirname(cfg.data.graph_file), exist_ok=True)
            torch.save((hetero_graph, id_maps), cfg.data.graph_file)
            print(f"💾 Hetero graph saved to {cfg.data.graph_file}")

    print(hetero_graph)
    print(hetero_graph.metadata())
    
    print("id_maps:")
    for k, v in id_maps.items():
        print(f"  {k}: {len(v)} entries")
        if len(v) <= 10:
            print(f"    {v}")
        else:
            print(f"    First 5: {dict(list(v.items())[:5])}")
    
    # -----------------------
    # Step 4.5: incorporate node features
    # -----------------------
    
    hetero_graph = attach_node_features(
        hetero_graph,
        id_maps,
        embeddings=None,  # could be path to precomputed embeddings
        emb_dim=cfg.model.embedding_dim
    )
    
    print("🔎 Node feature dimensions per node type:")
    for node_type in hetero_graph.node_types:
        x = hetero_graph[node_type].x
        print(f"  {node_type}: {x.shape if x is not None else 'No features'}")
    

    # -----------------------
    # Step 5: Build datasets
    # -----------------------
    print("✅ Building datasets...")
    train_ds = InteractionDataset(train_df, user_map, item_map,
                                  num_neg=cfg.train.num_neg, dynamic=True,
                                  all_interactions=train_interactions)
    print(f"   - train: {len(train_ds)} samples ({len(train_df)} positive)")
    valid_ds = InteractionDataset(valid_df, user_map, item_map,
                                 exhaustive_eval=False, num_eval_negs=cfg.eval.num_eval_negs,
                                 all_interactions=train_interactions)
    print(f"   - valid: {len(valid_ds)} samples ({len(valid_df)} positive)")
    test_ds = InteractionDataset(test_df, user_map, item_map,
                                 exhaustive_eval=True,
                                 all_interactions=train_interactions)
    print(f"   - test:  {len(test_ds)} samples ({len(test_df)} positive)")
    # === Build loaders ===
    if cfg.model.name == "ncf":
        train_loader = train_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=True)
        valid_loader = valid_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=False)
        test_loader  = test_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=False)

    else:  # Graph pipeline
        train_loader = train_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=True)
        valid_loader = valid_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=False)
        test_loader  = test_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=False)
        
    # === Sanity check on one batch per loader ===
    def check_loader(loader, name, etype):
        batch = next(iter(loader))
        assert hasattr(batch, "x_dict"), f"{name} missing x_dict"
        assert hasattr(batch, "edge_index_dict"), f"{name} missing edge_index_dict"
        assert etype in batch.edge_types, f"{name} missing edge type {etype}"
        assert hasattr(batch[etype], "edge_label_index"), f"{name} batch missing edge_label_index"
        assert hasattr(batch[etype], "edge_label"), f"{name} batch missing edge_label"
        print(f"✅ {name} loader check passed: {etype}, "
            f"{batch[etype].edge_label_index.shape[1]} supervision pairs")

    supervision_etype = (
        cfg.model.supervision_src_type,
        cfg.model.supervision_relation_type,
        cfg.model.supervision_dst_type,
    )

    check_loader(train_loader, "Train", supervision_etype)
    check_loader(valid_loader, "Valid", supervision_etype)
    check_loader(test_loader, "Test", supervision_etype)
        
    model = initialise_model(cfg, user_map=user_map, item_map=item_map, hetero_data=hetero_graph)
    
    # print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for name, p in model.named_parameters():
    #     print(name, p.shape, p.requires_grad)
    # -----------------------
    # Step 5: Dynamic monitor
    # -----------------------
    if cfg.model.name.lower() == "ncf":
        lightning_model = NCFRecLightning(
            model=model, lr=cfg.train.lr, k=cfg.eval.topk,
            loss_type=cfg.model.loss_type,
        )
    else:  # graph-based
        lightning_model = GraphRecLightning(
            model=model, lr=cfg.train.lr,
            supervision_src_type=cfg.model.supervision_src_type,
            supervision_relation_type=cfg.model.supervision_relation_type,
            supervision_dst_type=cfg.model.supervision_dst_type,
            k=cfg.eval.topk,
            loss_type=cfg.model.loss_type,
        )

    # -----------------------
    # Step 6: Train
    # -----------------------
    trainer, checkpoint_cb = initialise_trainer(cfg, run_dir)
    trainer.fit(lightning_model, train_loader, valid_loader)

    # -----------------------
    # Step 7: Reload best model
    # -----------------------
    best_model_path = checkpoint_cb.best_model_path
    print(f"✅ Best model saved at: {best_model_path}")

    if cfg.model.name.lower() == "ncf":
        best_model = NCFRecLightning.load_from_checkpoint(
            best_model_path,
            model=model,
            lr=cfg.train.lr,
            k=cfg.eval.topk,
            loss_type=cfg.model.loss_type,
        )
    else:
        best_model = GraphRecLightning.load_from_checkpoint(
            best_model_path,
            model=model,
            lr=cfg.train.lr,
            k=cfg.eval.topk,
            loss_type=cfg.model.loss_type,
        )

    best_model = best_model.to(trainer.lightning_module.device)

    # -----------------------
    # Step 8: Collect predictions
    # -----------------------
    serialise_predictions(trainer, best_model, valid_loader, run_dir, user_map, item_map, stage="val", is_graph=(cfg.model.name!="ncf"))
    serialise_predictions(trainer, best_model, test_loader, run_dir, user_map, item_map, stage="test", is_graph=(cfg.model.name!="ncf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    main(cfg)
