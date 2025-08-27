import os
import pandas as pd
import numpy as np
import yaml
from glob import glob
import traceback


REQUIRED_EDGE_COLS = ["source", "target", "relation", "datasourceId", "score", "year"]
YEAR_PRIORITY = ["curationYear", "studyYear", "publicationYear", "studyStartDate"]


class BaseParser:
    def __init__(self, root_dir: str, schema_file: str, output_dir: str, node_store: None):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.node_store = node_store or {}  # used by EdgeParser validation
        with open(schema_file, "r") as f:
            self.schema = yaml.safe_load(f)
        os.makedirs(self.output_dir, exist_ok=True)

    def deserialise(self, parquet_file: str) -> pd.DataFrame:
        return pd.read_parquet(parquet_file)

    def serialise(self, df: pd.DataFrame, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"💾 Saved → {out_path} ({len(df)} rows)")

    def parse(self):
        """
        Parse all schema-defined sources into one parquet per source.
        - If schema entry is a dict → single spec
        - If schema entry is a list → multiple specs (relations) for same source
        """
        all_data = {}

        for name, spec in self.schema.items():
            print(f"📦 Parsing: {name}")
            subdir_path = os.path.join(self.root_dir, name)
            if not os.path.exists(subdir_path):
                print(f"⚠️ No directory for {name}, skipping")
                continue

            # normalise spec to list
            specs = spec if isinstance(spec, list) else [spec]

            dfs = []
            num_rows = 0
            for pq in glob(os.path.join(subdir_path, "*.parquet")):
                try:
                    df = self.deserialise(pq)
                    num_rows += len(df)
                    # print(f"📄 Read {pq}: {len(df)} rows")
                    for sub_spec in specs:
                        try:
                            df_sub = self.apply_spec(df.copy(), sub_spec, name)
                            df_sub = self.validate(df_sub, sub_spec, name)

                            dfs.append(df_sub)
                        except Exception as e:
                            print(f"⚠️ Error applying spec for {sub_spec}: {e}")
                            traceback.print_exc()
                    # print(f"📦 Finished parsing: {name} ({num_rows} rows total)")
                except Exception as e:
                    print(f"⚠️ Error reading {pq}: {e}")
                    break

            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)

                out_name = self.output_name(name, spec)
                if out_name in all_data:
                    before = len(all_data[out_name])
                    all_data[out_name] = pd.concat([all_data[out_name], df_all], ignore_index=True)
                    after = len(all_data[out_name])
                    print(f"🔗 Merged {name} into {out_name}: {before} → {after} rows")
                else:
                    all_data[out_name] = df_all

        # 🔹 Serialise once per unique output name
        for out_name, df in all_data.items():
            out_path = os.path.join(self.output_dir, f"{out_name}.parquet")
            self.serialise(df, out_path)

        return all_data

    # Must be implemented by child
    def apply_spec(self, df, spec, name): 
        raise NotImplementedError

    # Must be implemented by child
    def output_name(self, name, spec):
        raise NotImplementedError
    
    # Default: no validation (override in EdgeParser)
    def validate(self, df, spec, name):
        return df
    
    @staticmethod
    def normalise(val):
        """
        Ensure a value is always a list:
        - None/NaN -> []
        - list/tuple -> unchanged
        - numpy.ndarray -> converted to list
        - scalar -> [val]
        """
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        # only call pd.isna on scalars
        try:
            if pd.isna(val):
                return []
        except Exception:
            pass
        return [val]



class NodeParser(BaseParser):
    def apply_spec(self, df, spec, name):
        cols = {"id": spec.get("id"), "name": spec.get("name")}
        if "props" in spec:
            for p in spec["props"]:
                if p in df.columns:
                    cols[p] = p
        cols = {k: v for k, v in cols.items() if v in df.columns}
        df = df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})
        # Ensure unique nodes based on 'id'
        df = df.drop_duplicates(subset=["id"])

        if name == "targets" and "biotype" in df.columns:
            df = df[df["biotype"] == "protein_coding"]
        
        return df

    def output_name(self, name, spec):
        return name  # node type
    
    def parse(self):
        node_dfs = super().parse()
        # build node_store = {node_type: set(ids)}
        node_store = {k: set(df["id"].astype(str)) for k, df in node_dfs.items() if "id" in df.columns}
        print("🔗 Node store built:")
        for k, v in node_store.items():
            print(f"  {k}: {len(v)} ids")
        return node_dfs, node_store


class EdgeParser(BaseParser):
    @staticmethod
    def _expand_targets(raw_val):
        """Ensure targets are iterable list"""
        if isinstance(raw_val, np.ndarray):
            raw_val = raw_val.tolist()
        if not isinstance(raw_val, list):
            raw_val = [raw_val]
        return raw_val

    @staticmethod
    def _add_props(edge, row, props):
        # Add all props first
        for p in props:
            if isinstance(p, str) and "=" in p and "constant:" in p:
                k, v = p.split("=", 1)
                edge[k] = float(v.split("constant:")[1]) if k == "score" else v.split("constant:")[1]
                continue
            if p in row and pd.notnull(row[p]):
                val = row[p]
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                if isinstance(val, (list, dict)):
                    val = str(val)
                edge[p] = val

        # Handle year
        if "year" in props:
            # Dynamic edges → pick best year candidate
            for col in YEAR_PRIORITY:
                if col in row and pd.notnull(row[col]):
                    if col == "studyStartDate":
                        try:
                            edge["year"] = pd.to_datetime(row[col], errors="coerce").year
                            return edge
                        except Exception:
                            continue
                    else:
                        edge["year"] = row[col]
                        return edge
            edge["year"] = np.nan  # fallback if no usable column
        else:
            # Static edges → force 0
            edge["year"] = 0

        return edge



    def apply_spec(self, df, spec, name):
        src_col = spec["source"]
        tgt_col = spec["target"]

        # Decide relation value
        if "relation" in spec:
            relation_value = spec["relation"]
            relation_is_column = True
        elif "relation_name" in spec:
            relation_value = spec["relation_name"]
            relation_is_column = False
        else:
            raise ValueError(f"No relation or relation_name in spec for {name}")

        props = spec.get("props", [])

        # === Case 1: Direct column-to-column edges ===
        if tgt_col in df.columns and "." not in tgt_col:
            sample_val = df[tgt_col].iloc[0]
            if not isinstance(sample_val, (list, tuple, np.ndarray)):
                expanded_edges = []
                for _, row in df.iterrows():
                    edge = {
                        "source": row[src_col],
                        "target": row[tgt_col],
                        "relation": row[relation_value] if relation_is_column else relation_value,
                    }
                    expanded_edges.append(self._add_props(edge, row, props))

                out = pd.DataFrame(expanded_edges)
                # if "year" not in out.columns:
                #     out["year"] = 0
                return out

        # === Case 2: Nested dict expansion (e.g. pathways.id) ===
        if "." in tgt_col:
            base_col, subfield = tgt_col.split(".", 1)
            if base_col in df.columns:
                expanded_edges = []
                for _, row in df.iterrows():
                    tgts = self._expand_targets(row.get(base_col, []))
                    for t in tgts:
                        if isinstance(t, dict):
                            t_val = t.get(subfield)
                            if not t_val:
                                continue
                            edge = {
                                "source": row[src_col],
                                "target": t_val,
                                "relation": row[relation_value] if relation_is_column else relation_value,
                            }
                            expanded_edges.append(self._add_props(edge, row, props))

                out = pd.DataFrame(expanded_edges)
                # if "year" not in out.columns:
                #     out["year"] = 0
                return out

        # === Case 3: List-like target expansion ===
        if tgt_col in df.columns:
            expanded_edges = []
            for _, row in df.iterrows():
                tgts = self._expand_targets(row.get(tgt_col, []))
                for t in tgts:
                    if not t:
                        continue
                    edge = {
                        "source": row[src_col],
                        "target": t,
                        "relation": row[relation_value] if relation_is_column else relation_value,
                    }
                    expanded_edges.append(self._add_props(edge, row, props))

            out = pd.DataFrame(expanded_edges)
            # if "year" not in out.columns:
            #     out["year"] = 0
            return out

        raise ValueError(f"Unsupported target {tgt_col} for {name}")


    def output_name(self, name, spec):
        return name  # one parquet per source dir
    
    def validate(self, df, spec, name):
        """Ensure sources/targets exist in node_store."""
        if not self.node_store:
            return df  # nothing to validate against

        src_valid = df["source"].astype(str).isin(set.union(*self.node_store.values()))
        tgt_valid = df["target"].astype(str).isin(set.union(*self.node_store.values()))

        df = df[src_valid & tgt_valid]
        return df
    
    def serialise(self, df, out_path):
        """
        Ensure required columns exist for explainability & temporal KG.
        Skip edges missing critical fields.
        """

        # Check required columns
        missing_cols = [c for c in REQUIRED_EDGE_COLS if c not in df.columns]
        if missing_cols:
            print(f"⚠️ Skipping save for {out_path}, missing columns: {missing_cols}")
            return

        before = len(df)

        # Find rows that would be dropped
        # dropped = df[df[REQUIRED_EDGE_COLS].isnull().any(axis=1)]
        # if not dropped.empty:
        #     print(f"⚠️ Dropping {len(dropped)} edges in {out_path} missing required fields:")
        #     if "id" in dropped.columns:
        #         print("   First 5 dropped IDs:", dropped["id"].head(5).tolist())
        #     else:
        #         cols_to_show = [c for c in ["source", "target", "relation"] if c in dropped.columns]
        #         print("   First 5 dropped rows:")
        #         print(dropped[cols_to_show].head(5).to_string(index=False))

        # Drop rows with missing required fields
        df = df.dropna(subset=REQUIRED_EDGE_COLS)

        # Normalise list-like props (e.g. literature)
        for col in df.columns:
            if col in ["literature"]:
                df[col] = df[col].apply(self.normalise)

        # Enforce schema ordering
        col_order = [c for c in REQUIRED_EDGE_COLS if c in df.columns] + \
                    [c for c in df.columns if c not in REQUIRED_EDGE_COLS]
        df = df[col_order]

        # Save parquet
        if not df.empty:
            df.to_parquet(out_path, index=False)
            print(f"💾 Saved → {out_path} ({len(df)} rows)")
        else:
            print(f"⚠️ No valid edges left for {out_path}, skipping save.")
