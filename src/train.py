#!/usr/bin/env python3
import argparse
import os
import datetime
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf


from src.data.split import supervision_edge_temporal_and_cold_split
from src.pipeline.build_hetero_graph import load_nodes, load_edges, get_most_evidented_edges, build_heterodata_with_cold_split
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

    nodes, id_to_type = load_nodes("data/kg_output/nodes/")
    edges = load_edges("data/kg_output/edges/")
    # this include all evidence edges before cutoff
    edges = edges[edges['year'] <= cfg.data.cutoff]
    # TODO: based on datatype or datasource
    edges = get_most_evidented_edges(edges)

    # TODO: create pretrained_embeddings
    pretrained_embeddings = None

    # -----------------------
    # Step 3: Generate id_maps
    # -----------------------
    user_map = {nid: i for i, nid in enumerate(nodes["diseases"]["id"].astype(str).tolist())}
    item_map = {nid: i for i, nid in enumerate(nodes["targets"]["id"].astype(str).tolist())}
    print(f"✅ Built id_maps: {len(user_map)} diseases, {len(item_map)} targets")
    all_interactions = build_all_interactions(train_df, user_map, item_map)
    # -----------------------
    # Step 4: Build hetero graph
    # -----------------------
    hetero_graph = build_heterodata_with_cold_split(nodes,
                                                    edges, 
                                                    train_df, 
                                                    valid_df, 
                                                    test_df, 
                                                    cfg.data.cutoff, 
                                                    cfg.data.horizon,
                                                    supervision_source=cfg.model.supervision_src_type, 
                                                    supervision_target=cfg.model.supervision_dst_type, 
                                                    supervision_relation=cfg.model.supervision_relation_type)

    print(hetero_graph)
    print(hetero_graph.metadata())

    # -----------------------
    # Step 5: Build datasets
    # -----------------------
    train_ds = InteractionDataset(train_df, user_map, item_map,
                                  num_neg=cfg.train.num_neg, dynamic=True,
                                  all_interactions=all_interactions)
    valid_ds = InteractionDataset(valid_df, user_map, item_map,
                                  exhaustive_eval=True,
                                  all_interactions=all_interactions)
    test_ds = InteractionDataset(test_df, user_map, item_map,
                                 exhaustive_eval=True,
                                 all_interactions=all_interactions)
    # === Build loaders ===
    if cfg.model.name == "ncf":
        train_loader = train_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=True)
        valid_loader = valid_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=False)
        test_loader  = test_ds.build_ncf_loader(batch_size=cfg.train.batch_size, shuffle=False)
        # TODO: integrate pretrained_embeddings
        model = initialise_model(cfg, user_map=user_map, item_map=item_map, pretrained_embeddings=pretrained_embeddings)

    else:  # Graph pipeline
        train_loader = train_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=True)
        valid_loader = valid_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=False)
        test_loader  = test_ds.build_graph_loader(hetero_graph, batch_size=cfg.train.batch_size, shuffle=False)
        model = initialise_model(cfg, user_map=user_map, item_map=item_map, hetero_data=hetero_graph, pretrained_embeddings=pretrained_embeddings)
        # TODO: integrate pretrained_embeddings
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
            model=model, lr=cfg.train.lr, k=cfg.eval.topk,
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
