import os
import csv
import re

def extract_ensembl_gene_id(xref_string):
    """
    Extracts Ensembl Gene ID (ENSG) from the Xref string.
    Example xref_string: '...|ensembl:ENSG00000099942.13(gene)|...'
    Returns the first found ENSG ID (without version) or None.
    """
    if not xref_string or xref_string == '-':
        return None
    
    # Regex to find ensembl:ENSG followed by digits
    # It might have version number .XX
    match = re.search(r'ensembl:(ENSG\d+)', xref_string)
    if match:
        return match.group(1)
    return None

def extract_pubmed_ids(pub_string):
    """
    Extracts PubMed IDs from the Publication Identifier string.
    Example pub_string: 'pubmed:10022120|mint:MINT-6731034'
    Returns a list of PubMed IDs.
    """
    if not pub_string or pub_string == '-':
        return []
    
    # Find all pubmed:XXXX entries
    matches = re.findall(r'pubmed:(\d+)', pub_string)
    return matches

def is_protein(type_string):
    """
    Checks if the interactor type is protein.
    Example type_string: 'psi-mi:"MI:0326"(protein)'
    """
    return 'psi-mi:"MI:0326"(protein)' in type_string

def parse_intact_file(file_path):
    """
    Parses the IntAct PSI-MITAB file and extracts protein-coding gene interactions
    along with literature references.
    
    Yields:
        dict: {
            "source_gene": str,
            "target_gene": str,
            "pubmed_ids": list[str]
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Verify header
        header = f.readline()
        if not header.startswith('#'):
            f.seek(0) # No header, reset (though MITAB usually has one)
        
        # Reader with tab delimiter
        # We process line by line to handle potential quoting issues manually if needed, 
        # but csv module is usually robust.
        # Use QUOTE_NONE to avoid issues with quotes inside fields (common in MITAB)
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        for line_idx, row in enumerate(reader):
            if not row: 
                continue
                
            # Ensure we have enough columns (at least up to 24 for Xrefs)
            if len(row) < 24:
                # print(f"Skipping line {line_idx}: Not enough columns ({len(row)})")
                continue
                
            # Columns (0-indexed):
            # 8: Publication Identifier(s)
            # 20: Type(s) interactor A
            # 21: Type(s) interactor B
            # 22: Xref(s) interactor A
            # 23: Xref(s) interactor B
            
            type_a = row[20]
            type_b = row[21]
            
            # Filter for protein-protein interactions
            if not (is_protein(type_a) and is_protein(type_b)):
                # print(f"Skipping line {line_idx}: Not protein-protein ({type_a}, {type_b})")
                continue
                
            xref_a = row[22]
            xref_b = row[23]
            
            gene_a = extract_ensembl_gene_id(xref_a)
            gene_b = extract_ensembl_gene_id(xref_b)
            
            # Filter for protein-coding genes (must have ENSG extracted)
            if not gene_a or not gene_b:
                # print(f"Skipping line {line_idx}: Missing ENSG ID. A: {gene_a}, B: {gene_b}")
                # print(f"  Xref A: {xref_a}")
                # print(f"  Xref B: {xref_b}")
                continue
            
            pub_string = row[8]
            pubmed_ids = extract_pubmed_ids(pub_string)
            
            yield {
                "source_gene": gene_a,
                "target_gene": gene_b,
                "pubmed_ids": pubmed_ids,
                "raw_source_id": row[0], # For debugging/reference
                "raw_target_id": row[1]
            }

if __name__ == "__main__":
    # Example usage (for testing)
    import sys
    
    # Default path for testing if not provided
    test_file = "/Users/pui.chungsiu/Documents/opentarget_het_graph/data/test_intact_human.txt"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        
    print(f"Parsing {test_file}...")
    try:
        count = 0
        for interaction in parse_intact_file(test_file):
            print(interaction)
            count += 1
            if count >= 5:
                print("...")
                break
        print(f"Parsed partial results shown above.")
    except Exception as e:
        print(f"Error: {e}")
