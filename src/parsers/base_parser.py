import os
import pandas as pd
from glob import glob
import yaml


class BaseParser:
    """Base class to read parquet files in subdirectories."""

    def __init__(self, root_dir: str, schema_file: str):
        self.root_dir = root_dir
        with open(schema_file, "r") as f:
            self.schema = yaml.safe_load(f)

    def load_dataframes(self) -> dict:
        """Read all parquet files under root_dir, grouped by subdirectory."""
        data = {}
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            parquet_files = glob(os.path.join(subdir_path, "*.parquet"))
            if not parquet_files:
                continue

            dfs = []
            for pq in parquet_files:
                try:
                    dfs.append(pd.read_parquet(pq))
                except Exception as e:
                    print(f"⚠️ Skipping {pq}, error: {e}")

            if dfs:
                data[subdir] = pd.concat(dfs, ignore_index=True)

        return data


class NodeParser(BaseParser):
    """Parse node parquet files according to a YAML schema."""

    def parse(self) -> pd.DataFrame:
        node_dfs = []
        data = self.load_dataframes()

        for node_type, df in data.items():
            if node_type not in self.schema:
                continue

            mapping = self.schema[node_type]
            parsed = pd.DataFrame()
            parsed["id"] = df[mapping["id"]]

            if "name" in mapping and mapping["name"] in df.columns:
                parsed["name"] = df[mapping["name"]]
            else:
                parsed["name"] = None

            parsed["type"] = node_type

            # Add extra attributes
            for col in mapping.get("extra", []):
                if col in df.columns:
                    parsed[col] = df[col]

            node_dfs.append(parsed)

        return pd.concat(node_dfs, ignore_index=True) if node_dfs else pd.DataFrame()


class EdgeParser(BaseParser):
    """Parse evidence parquet files into edges according to a YAML schema."""

    def parse(self, valid_nodes: pd.DataFrame = None) -> pd.DataFrame:
        edge_dfs = []
        data = self.load_dataframes()

        for source, df in data.items():
            if source not in self.schema:
                continue

            mapping = self.schema[source]
            parsed = pd.DataFrame()
            parsed["source_id"] = df[mapping["source"]]
            parsed["target_id"] = df[mapping["target"]]
            parsed["relation"] = mapping.get("relation", source)

            # Add properties
            for col in mapping.get("props", []):
                if col in df.columns:
                    parsed[col] = df[col]

            parsed["evidence_source"] = source

            # ✅ Validation: check source_id & target_id exist in valid_nodes
            if valid_nodes is not None:
                valid_ids = set(valid_nodes["id"].unique())
                before = len(parsed)

                parsed = parsed[
                    parsed["source_id"].isin(valid_ids) & parsed["target_id"].isin(valid_ids)
                ]

                after = len(parsed)
                if before != after:
                    print(f"⚠️ {before - after} edges from {source} dropped (missing nodes).")

            edge_dfs.append(parsed)

        return pd.concat(edge_dfs, ignore_index=True) if edge_dfs else pd.DataFrame()
