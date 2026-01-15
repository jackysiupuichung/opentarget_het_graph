import os
import csv
import re
import pandas as pd
from src.parsers.parser import EdgeParser

class IntActParser(EdgeParser):
    def __init__(self, root_dir, schema_file, output_dir, node_store=None):
        super().__init__(root_dir, schema_file, output_dir, node_store)
    
    def extract_ensembl_gene_id(self, xref_string):
        if not xref_string or xref_string == '-':
            return None
        match = re.search(r'ensembl:(ENSG\d+)', xref_string)
        return match.group(1) if match else None

    def extract_pubmed_ids(self, pub_string):
        if not pub_string or pub_string == '-':
            return []
        matches = re.findall(r'pubmed:(\d+)', pub_string)
        return list(set(matches))

    def is_protein(self, type_string):
        return 'psi-mi:"MI:0326"(protein)' in type_string

    def parse(self, file_path=None):
        """
        Custom parse for IntAct MITAB format.
        """
        if not file_path:
            intact_dir = os.path.join(self.root_dir, "intact")
            files = [os.path.join(intact_dir, f) for f in os.listdir(intact_dir) if f.endswith(".txt")]
            if not files:
                print(f"⚠️ No IntAct files found in {intact_dir}")
                return {}
            file_path = files[0]

        print(f"📦 Parsing IntAct: {file_path}")
        results = []
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            if not header.startswith('#'): f.seek(0)
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            
            for row in reader:
                if not row or len(row) < 24: continue
                if not (self.is_protein(row[20]) and self.is_protein(row[21])): continue
                gene_a = self.extract_ensembl_gene_id(row[22])
                gene_b = self.extract_ensembl_gene_id(row[23])
                if not gene_a or not gene_b: continue
                
                pubmed_ids = self.extract_pubmed_ids(row[8])
                interaction = {
                    "sourceId": gene_a,
                    "targetId": gene_b,
                    "source_type": "target",
                    "target_type": "target",
                    "relation": "interacts_with",
                    "datasourceId": "intact",
                    "score": 1.0,
                    "year": None, # Will be filled by serialise
                    "literature": pubmed_ids
                }
                results.append(interaction)

        df = pd.DataFrame(results)
        df = self.validate(df, None, "intact")
        
        # Determine output name dynamically
        out_name = self.output_name("intact", {"relation": "interacts_with", "props": ["datasourceId=constant:intact", "source_type=constant:target", "target_type=constant:target"]}, df)
        out_path = os.path.join(self.output_dir, f"{out_name}.parquet")
        self.serialise(df, out_path)
        return {"intact": df}

if __name__ == "__main__":
    test_file = "/Users/pui.chungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/intact/test_intact_human.txt"
    parser = IntActParser(root_dir="data", schema_file="config/edge_schema.yaml", output_dir="output")
    parser.parse(file_path=test_file)
