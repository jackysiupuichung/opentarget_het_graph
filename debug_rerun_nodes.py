
from src.parsers.parser import NodeParser
import argparse

def main():
    # Hardcoded paths based on project structure
    input_dir = "data/evidenceDated_subset/23.06"
    schema_file = "config/node_schema.yaml"
    output_dir = "output/nodes"
    
    print("🔹 Re-running Node Parsing...")
    node_parser = NodeParser(input_dir, schema_file, output_dir, node_store=None)
    node_parser.parse()
    print("✅ Nodes updated.")

if __name__ == "__main__":
    main()
