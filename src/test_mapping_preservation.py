
import unittest
import pandas as pd
import torch
import shutil
import tempfile
import os
from pathlib import Path
from pipeline.build_event_graph import build_hetero_graph, build_event_graph

class TestMappingPreservation(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        data = {
            'sourceId': ['n1', 'n2', 'n1', 'n3'],
            'targetId': ['t1', 't1', 't2', 't2'],
            'source_type': ['gene', 'gene', 'gene', 'drug'],
            'target_type': ['target', 'target', 'target', 'target'],
            'relation': ['binds', 'binds', 'activates', 'inhibits'],
            'datasourceId': ['db1', 'db1', 'db2', 'db3'],
            'edge_time': [2020, 2021, 2022, 2023],
            'edge_weight': [0.1, 0.2, 0.3, 0.4]
        }
        self.df = pd.DataFrame(data)
        self.test_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.test_dir, 'events.parquet')
        self.df.to_parquet(self.parquet_path)
        self.output_path = os.path.join(self.test_dir, 'graph.pt')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_build_hetero_graph_mappings(self):
        hetero_data, mappings = build_hetero_graph(self.df)
        
        # Check keys
        self.assertIn('node_mapping', mappings)
        self.assertIn('node_type_mapping', mappings)
        self.assertIn('edge_type_mapping', mappings)
        self.assertIn('edge_mapping', mappings)
        
        # Verify Node Mapping
        node_mapping = mappings['node_mapping']
        self.assertIn('gene', node_mapping)
        self.assertIn('n1', node_mapping['gene'])
        self.assertEqual(node_mapping['gene']['n1'], 0) # Sorted: n1, n2, n3 -> n1=0
        
        # Verify Edge Mapping
        edge_mapping = mappings['edge_mapping']
        # binds::db1 event (indices 0 and 1)
        edge_key = "('gene', 'binds::db1', 'target')"
        self.assertTrue(torch.equal(edge_mapping[edge_key], torch.tensor([0, 1])))
        
        # activates::db2 event (index 2)
        edge_key_2 = "('gene', 'activates::db2', 'target')"
        self.assertTrue(torch.equal(edge_mapping[edge_key_2], torch.tensor([2])))

    def test_build_event_graph_file_output(self):
        build_event_graph(self.parquet_path, self.output_path)
        
        # Check files exist
        self.assertTrue(os.path.exists(self.output_path))
        mapping_path = self.output_path.replace(".pt", "_mappings.pt")
        self.assertTrue(os.path.exists(mapping_path))
        
        # Load and verify
        mappings = torch.load(mapping_path)
        self.assertIn('node_mapping', mappings)

if __name__ == '__main__':
    unittest.main()
