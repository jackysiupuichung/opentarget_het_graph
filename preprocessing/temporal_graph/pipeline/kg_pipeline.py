import os
import sys
import argparse
import pandas as pd

# Add the parent directory to path to allow importing parsers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.parser import NodeParser, EdgeParser
from parsers.intact.parser import IntActParser
from parsers.go_ontology.parser import GOOntologyParser


def run_pipeline(input, node_input, node_schema, edge_schema, static_edge_schema, node_output, edge_output, static_edge_output, debug=False):
    if debug:
        print("🐛 [DEBUG MODE] Nodes: all files. Edges: 1 file per datasource subdirectory.")
    print("🔹 Parsing nodes...")
    node_parser = NodeParser(node_input, node_schema, node_output, node_store=None)
    node_data, node_store = node_parser.parse()

    print("🔹 Parsing edges...")
    edge_parser = EdgeParser(input, edge_schema, edge_output, node_store=node_store, debug=debug)
    edge_parser.parse()

    # print("🔹 Parsing intact edges...")
    # intact_parser = IntActParser(input, edge_schema, edge_output, node_store=node_store)
    # intact_parser.parse()

    print("🔹 Parsing go ontology edges...")
    go_ontology_parser = GOOntologyParser(node_input, static_edge_schema, static_edge_output, node_store=node_store, static=True)
    go_ontology_parser.parse()

    print("🔹 Parsing static edges...")
    static_edge_parser = EdgeParser(input, static_edge_schema, static_edge_output, node_store=node_store, static=True)
    static_edge_parser.parse()

    print("✅ Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Construction Pipeline")

    parser.add_argument("--input", required=True, help="Directory with evidence parquet files (edge source)")
    parser.add_argument("--node-input", default=None, help="Directory with node parquet files; defaults to --input if omitted")
    parser.add_argument("--node-schema", required=True, help="YAML schema for nodes")
    parser.add_argument("--ot-version", choices=["23.06", "26.03"], default="26.03",
                        help="OpenTargets release; selects config/edge_schema_<ver>.yaml unless --edge-schema is given")
    parser.add_argument("--edge-schema", default=None, help="Override edge schema path (otherwise derived from --ot-version)")
    parser.add_argument("--static-edge-schema", required=True, help="YAML schema for static edges")
    parser.add_argument("--node-output", required=True, help="Output directory for parsed node parquet files")
    parser.add_argument("--edge-output", required=True, help="Output directory for parsed edge parquet files")
    parser.add_argument("--static-edge-output", required=True, help="Output directory for parsed static edge parquet files")
    parser.add_argument("--debug", action="store_true", help="Debug mode: read only 1 file per datasource subdirectory")

    args = parser.parse_args()

    edge_schema = args.edge_schema or f"config/edge_schema_{args.ot_version}.yaml"
    if not os.path.exists(edge_schema):
        raise FileNotFoundError(f"Edge schema not found: {edge_schema}")
    print(f"🔹 Using edge schema: {edge_schema} (OT {args.ot_version})")

    run_pipeline(
        input=args.input,
        node_input=args.node_input if args.node_input else args.input,
        node_schema=args.node_schema,
        edge_schema=edge_schema,
        static_edge_schema=args.static_edge_schema,
        node_output=args.node_output,
        edge_output=args.edge_output,
        static_edge_output=args.static_edge_output,
        debug=args.debug,
    )
