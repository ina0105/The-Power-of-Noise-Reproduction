#!/usr/bin/env python3
"""
Script to extract examples from data/gen_res_max_length_prompt directory in the same format as examples_11doc_better_than_1doc.txt
"""

import json
import re
import os
from typing import List, Dict, Optional

def load_json_file(directory: str, filename: str) -> List[Dict]:
    """Loads data from a JSON file."""
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert the JSON structure to a list of dictionaries
    result = []
    if isinstance(data, list) and len(data) > 0:
        # Check if it's the extended format with nested lists
        if isinstance(data[0], dict) and 'example_id' in data[0]:
            # Already in the right format
            return data
        else:
            # It's the batch format - need to expand
            for batch in data:
                if isinstance(batch, dict) and 'example_id' in batch:
                    num_examples = len(batch['example_id'])
                    for i in range(num_examples):
                        example = {
                            'example_id': batch['example_id'][i],
                            'query': batch['query'][i],
                            'prompt': batch['prompt'][i],
                            'generated_answer': batch['generated_answer'][i],
                            'gold_document_idx': batch['gold_document_idx'][i],
                            'document_indices': batch['document_indices'][i] if 'document_indices' in batch else []
                        }
                        result.append(example)
    
    return result

def extract_example_ids_from_file(filepath: str) -> List[int]:
    """Extract example IDs from the existing examples file."""
    example_ids = []
    with open(filepath, 'r') as f:
        content = f.read()
        # Find all "Example ID: <number>" patterns
        pattern = re.compile(r'Example ID: (-?\d+)')
        matches = pattern.findall(content)
        example_ids = [int(match) for match in matches]
    return example_ids

def get_gold_document_from_prompt(prompt: str) -> Optional[str]:
    """Extract the gold document from the prompt by finding the document that contains the answer."""
    # The prompt format is: Documents:\nDocument [idx](Title: ...) ...\nQuestion: ...
    # We need to find which document is the gold one
    # For now, we'll extract all documents and return the first one (since in 1 doc case it's the gold)
    # For 11 doc, we'll need to identify which one is gold based on document indices
    
    # Split by "Document [" to get individual documents
    if "Documents:\n" not in prompt:
        return None
    
    documents_section = prompt.split("Documents:\n")[1].split("\nQuestion:")[0]
    documents = documents_section.split("Document [")
    
    # The first document (after Documents:\n) is the gold one for 1 doc case
    if len(documents) > 1:
        # Remove the "Document [" prefix and get the content
        doc_content = documents[1]  # First actual document
        # Find the closing bracket and extract title and content
        if "](Title: " in doc_content:
            # Extract everything after the title
            parts = doc_content.split("](Title: ", 1)
            if len(parts) > 1:
                title_and_content = parts[1]
                # Find where title ends (first closing parenthesis)
                title_end = title_and_content.find(") ")
                if title_end != -1:
                    title = title_and_content[:title_end]
                    content = title_and_content[title_end + 2:]
                    # Reconstruct the document format
                    doc_idx_match = re.search(r'^(\d+)', parts[0])
                    doc_idx = doc_idx_match.group(1) if doc_idx_match else "?"
                    return f"Document [{doc_idx}](Title: {title}) {content}"
    
    return None

def find_gold_document_in_prompt(prompt: str, gold_doc_idx: int) -> Optional[str]:
    """Find the specific gold document in a multi-document prompt."""
    if "Documents:\n" not in prompt:
        return None
    
    documents_section = prompt.split("Documents:\n")[1].split("\nQuestion:")[0]
    
    # Find document with the matching index
    pattern = re.compile(rf'Document \[{gold_doc_idx}\]\(Title: ([^)]+)\) (.+?)(?=Document \[|\Z)', re.DOTALL)
    match = pattern.search(documents_section)
    
    if match:
        title = match.group(1)
        content = match.group(2).strip()
        return f"Document [{gold_doc_idx}](Title: {title}) {content}"
    
    return None

def main():
    # Paths
    examples_file = "/home/scur2158/The-Power-of-Noise/examples_11doc_better_than_1doc.txt"
    output_file = "/home/scur2158/The-Power-of-Noise/examples_max_length_prompt.txt"
    
    # Data directories
    base_dir = "/home/scur2158/The-Power-of-Noise/data/gen_res_max_length_prompt"
    llm_folder = "Llama-2-7b-chat-hf"
    split = "train"
    prompt_type = "classic"
    retriever = "contriever"
    
    dir_1doc = f"{base_dir}/{llm_folder}/{split}/{prompt_type}/{retriever}/1_doc"
    dir_11doc = f"{base_dir}/{llm_folder}/{split}/{prompt_type}/{retriever}/11_doc"
    
    # Load dataset for ground truth
    dataset_path = "/home/scur2158/The-Power-of-Noise/data/10k_train_dataset.json"
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    dataset_dict = {item['example_id']: item for item in dataset}
    
    # Load data from JSON files
    print("Loading 1 doc data...")
    json_file_1doc = "numdoc1_gold_at0_rand_answerless_info_all_extended.json"
    data_1doc = load_json_file(dir_1doc, json_file_1doc)
    print(f"Loaded {len(data_1doc)} examples from 1 doc")
    
    print("Loading 11 doc data (near scenario)...")
    json_file_11doc = "numdoc11_gold_at10_rand_answerless_info_all_extended.json"
    data_11doc = load_json_file(dir_11doc, json_file_11doc)
    print(f"Loaded {len(data_11doc)} examples from 11 doc")
    
    # Create lookup dictionaries
    data_1doc_dict = {}
    for example in data_1doc:
        example_id = int(example['example_id'])
        data_1doc_dict[example_id] = {
            'query': example['query'],
            'prompt': example['prompt'],
            'generated_answer': example['generated_answer'],
            'gold_document_idx': example['gold_document_idx'],
            'document_indices': example['document_indices'] if 'document_indices' in example else []
        }
    
    data_11doc_dict = {}
    for example in data_11doc:
        example_id = int(example['example_id'])
        data_11doc_dict[example_id] = {
            'query': example['query'],
            'prompt': example['prompt'],
            'generated_answer': example['generated_answer'],
            'gold_document_idx': example['gold_document_idx'],
            'document_indices': example['document_indices'] if 'document_indices' in example else []
        }
    
    # Extract example IDs from the existing file (if it exists)
    example_ids_from_file = []
    if os.path.exists(examples_file):
        example_ids_from_file = extract_example_ids_from_file(examples_file)
        print(f"Found {len(example_ids_from_file)} example IDs in the original file")
    
    # Find example IDs that exist in both datasets
    common_ids = sorted(set(data_1doc_dict.keys()) & set(data_11doc_dict.keys()))
    print(f"Found {len(common_ids)} example IDs that exist in both datasets")
    
    # Use example IDs from file if available, otherwise use all common IDs
    if example_ids_from_file:
        example_ids = [eid for eid in example_ids_from_file if eid in common_ids]
        print(f"Using {len(example_ids)} example IDs from original file that exist in both datasets")
    else:
        example_ids = common_ids
        print(f"Using all {len(example_ids)} common example IDs")
    
    # Write output file
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("EXAMPLES FROM data/gen_res_max_length_prompt\n")
        f.write(f"Total examples found: {len(example_ids)}\n")
        f.write("=" * 120 + "\n\n")
        
        found_count = 0
        for idx, example_id in enumerate(example_ids):
            if example_id not in data_1doc_dict or example_id not in data_11doc_dict:
                continue
            
            if example_id not in dataset_dict:
                continue
            
            found_count += 1
            data_1 = data_1doc_dict[example_id]
            data_11 = data_11doc_dict[example_id]
            dataset_item = dataset_dict[example_id]
            
            query = data_1['query']
            ground_truth = dataset_item['answers']
            
            # Get gold document
            gold_doc_1 = get_gold_document_from_prompt(data_1['prompt'])
            gold_doc_11 = find_gold_document_in_prompt(data_11['prompt'], int(data_11['gold_document_idx']))
            gold_doc = gold_doc_11 if gold_doc_11 else gold_doc_1
            
            # Check if answers match
            def normalize_answer(ans):
                return ans.lower().strip()
            
            def check_match(pred, truths):
                pred_norm = normalize_answer(pred)
                for truth in truths:
                    if normalize_answer(truth) in pred_norm or pred_norm in normalize_answer(truth):
                        return True
                return False
            
            match_1 = check_match(data_1['generated_answer'], ground_truth)
            match_11 = check_match(data_11['generated_answer'], ground_truth)
            
            f.write("=" * 120 + "\n")
            f.write(f"EXAMPLE {found_count} (Index {idx+1}, Example ID: {example_id})\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"GROUND TRUTH: {ground_truth}\n\n")
            
            if gold_doc:
                f.write("-" * 120 + "\n")
                f.write("GOLD DOCUMENT:\n")
                f.write("-" * 120 + "\n")
                f.write(f"{gold_doc}\n\n")
            
            f.write("-" * 120 + "\n")
            status_1 = "CORRECT" if match_1 else "WRONG"
            f.write(f"1 DOCUMENT - {status_1}:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{data_1['generated_answer']}\n\n")
            
            f.write("-" * 120 + "\n")
            status_11 = "CORRECT" if match_11 else "WRONG"
            f.write(f"11 DOCUMENTS - {status_11}:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{data_11['generated_answer']}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("FULL PROMPT (1 DOC) - Complete untrimmed:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{data_1['prompt']}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("FULL PROMPT (11 DOC) - Complete untrimmed:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{data_11['prompt']}\n\n")
            
            f.write("\n")
    
    print(f"Extracted {found_count} examples to {output_file}")

if __name__ == "__main__":
    main()



