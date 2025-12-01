#!/usr/bin/env python3
"""
Generate examples file comparing 13 docs vs 1 doc for gen_res_prompt_ans_only experiment.
Format similar to examples_max_length_prompt.txt
"""

import json
import os
from typing import List, Dict

def load_results(filepath: str) -> List[Dict]:
    """Load extended results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_answer(text: str) -> str:
    """Simple normalization for comparison."""
    return text.lower().strip()

def is_answer_correct(generated_answer: str, ground_truth_answers: List[str]) -> bool:
    """Check if generated answer matches any ground truth."""
    normalized_gen = normalize_answer(generated_answer)
    for gt in ground_truth_answers:
        normalized_gt = normalize_answer(gt)
        if normalized_gt in normalized_gen or normalized_gen in normalized_gt:
            return True
    return False

def find_gold_document(prompt: str, gold_doc_idx: int) -> str:
    """Extract the gold document from the prompt."""
    # Look for the document with the matching index
    lines = prompt.split('\n')
    in_documents = False
    gold_doc_lines = []
    
    for line in lines:
        if line.startswith('Documents:'):
            in_documents = True
            continue
        if line.startswith('Question:'):
            break
        if in_documents and f'[{gold_doc_idx}]' in line:
            # Found the gold document, collect it
            gold_doc_lines.append(line)
            # Continue until next Document or Question
            continue
        if in_documents and line.startswith('Document ['):
            if gold_doc_lines:
                break  # We've moved to next document
            continue
    
    return '\n'.join(gold_doc_lines) if gold_doc_lines else "Gold document not found in prompt"

def generate_examples_file(results_1doc: List[Dict], results_13doc: List[Dict], 
                          output_file: str = 'examples_ans_only_13_vs_1.txt', 
                          max_examples: int = 20):
    """Generate examples file in the same format as examples_max_length_prompt.txt"""
    
    # Create lookup dictionaries
    results_1doc_dict = {r['example_id']: r for r in results_1doc}
    results_13doc_dict = {r['example_id']: r for r in results_13doc}
    
    # Find cases where 13 doc is correct but 1 doc is incorrect
    mismatches = []
    for example_id in results_1doc_dict.keys():
        if example_id not in results_13doc_dict:
            continue
        
        r1 = results_1doc_dict[example_id]
        r13 = results_13doc_dict[example_id]
        
        correct_1doc = r1['ans_match_after_norm']
        correct_13doc = r13['ans_match_after_norm']
        
        if not correct_1doc and correct_13doc:
            mismatches.append({
                'example_id': example_id,
                'query': r1['query'],
                'ground_truth': r1['answers'],
                'generated_1doc': r1['generated_answer'],
                'generated_13doc': r13['generated_answer'],
                'prompt_1doc': r1['prompt'],
                'prompt_13doc': r13['prompt'],
                'gold_doc_idx': r1['gold_document_idx']
            })
    
    # Limit to max_examples
    mismatches = mismatches[:max_examples]
    
    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write(f"EXAMPLES FROM data/gen_res_prompt_ans_only\n")
        f.write(f"Total examples found: {len(mismatches)}\n")
        f.write("=" * 120 + "\n\n")
        
        for idx, ex in enumerate(mismatches, 1):
            f.write("=" * 120 + "\n")
            f.write(f"EXAMPLE {idx} (Index {idx}, Example ID: {ex['example_id']})\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"QUERY: {ex['query']}\n")
            f.write(f"GROUND TRUTH: {ex['ground_truth']}\n\n")
            
            # Extract gold document
            gold_doc = find_gold_document(ex['prompt_1doc'], int(ex['gold_doc_idx']))
            
            f.write("-" * 120 + "\n")
            f.write("GOLD DOCUMENT:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{gold_doc}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("1 DOCUMENT - INCORRECT:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{ex['generated_1doc']}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("13 DOCUMENTS - CORRECT:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{ex['generated_13doc']}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("FULL PROMPT (1 DOC) - Complete untrimmed:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{ex['prompt_1doc']}\n\n")
            
            f.write("-" * 120 + "\n")
            f.write("FULL PROMPT (13 DOC) - Complete untrimmed:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{ex['prompt_13doc']}\n\n")
    
    print(f"Examples file created: {output_file}")
    print(f"Total examples written: {len(mismatches)}")

def main():
    base_dir = "data/gen_res_prompt_ans_only/Llama-2-7b-chat-hf/train/classic/contriever"
    
    file_1doc = os.path.join(base_dir, "1_doc/numdoc1_gold_at0_rand_answerless_info_all_extended.json")
    file_13doc = os.path.join(base_dir, "13_doc/numdoc13_gold_at12_rand_answerless_info_all_extended.json")
    
    if not os.path.exists(file_1doc):
        print(f"Error: File not found: {file_1doc}")
        return
    
    if not os.path.exists(file_13doc):
        print(f"Error: File not found: {file_13doc}")
        return
    
    print("Loading results...")
    results_1doc = load_results(file_1doc)
    results_13doc = load_results(file_13doc)
    
    print(f"Loaded {len(results_1doc)} results from 1 doc")
    print(f"Loaded {len(results_13doc)} results from 13 doc")
    
    print("Generating examples file...")
    generate_examples_file(results_1doc, results_13doc, 'examples_ans_only_13_vs_1.txt', max_examples=20)

if __name__ == "__main__":
    main()



