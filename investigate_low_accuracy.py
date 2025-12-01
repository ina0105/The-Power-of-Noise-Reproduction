#!/usr/bin/env python3
"""Investigate low accuracy issue with Llama 3 results"""

import json
import os

file_path = "data/gen_res_prompt_ans_only_llama3_unquantized/Meta-Llama-3-8B-Instruct/train/classic/contriever/1_doc/numdoc1_gold_at0_rand_answerless_info_all_extended.json"

print("Loading results...")
with open(file_path, 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")

# Count matches
matches = sum(1 for ex in data if ex.get('ans_match_after_norm', False))
print(f"Matches: {matches}")
print(f"Accuracy: {matches/len(data):.4f}")

# Check first 10 examples
print("\n" + "="*80)
print("First 10 examples:")
print("="*80)

for i in range(min(10, len(data))):
    ex = data[i]
    print(f"\nExample {i+1}:")
    print(f"  Query: {ex['query'][:150]}")
    print(f"  Generated answer: {repr(ex['generated_answer'][:200])}")
    print(f"  Ground truth: {ex['answers']}")
    print(f"  Match: {ex.get('ans_match_after_norm', False)}")
    print(f"  Answer in docs: {ex.get('ans_in_documents', False)}")
    
    # Check if generated answer contains common issues
    gen = ex['generated_answer'].lower()
    if 'no-res' in gen or 'no res' in gen:
        print("  ⚠️  Contains NO-RES")
    if len(ex['generated_answer'].strip()) == 0:
        print("  ⚠️  Empty answer")
    if len(ex['generated_answer']) > 500:
        print(f"  ⚠️  Very long answer ({len(ex['generated_answer'])} chars)")

# Check for common patterns
print("\n" + "="*80)
print("Pattern Analysis:")
print("="*80)

empty_answers = sum(1 for ex in data if not ex['generated_answer'].strip())
no_res_answers = sum(1 for ex in data if 'no-res' in ex['generated_answer'].lower() or 'no res' in ex['generated_answer'].lower())
very_long = sum(1 for ex in data if len(ex['generated_answer']) > 500)

print(f"Empty answers: {empty_answers} ({100*empty_answers/len(data):.2f}%)")
print(f"NO-RES answers: {no_res_answers} ({100*no_res_answers/len(data):.2f}%)")
print(f"Very long answers (>500 chars): {very_long} ({100*very_long/len(data):.2f}%)")

# Check answer extraction - look at raw output
print("\n" + "="*80)
print("Checking answer extraction (first 3 examples with full prompt):")
print("="*80)

# Load the pickle file to see raw outputs
import pickle
pkl_file = "data/gen_res_prompt_ans_only_llama3_unquantized/Meta-Llama-3-8B-Instruct/train/classic/contriever/1_doc/numdoc1_gold_at0_rand_answerless_info_250.pkl"

if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)
    
    if pkl_data and len(pkl_data) > 0:
        first_batch = pkl_data[0]
        if 'generated_answer' in first_batch and len(first_batch['generated_answer']) > 0:
            print(f"\nFirst generated answer from pickle (raw):")
            print(repr(first_batch['generated_answer'][0][:500]))
            
            # Check the prompt to see answer marker
            if 'prompt' in first_batch and len(first_batch['prompt']) > 0:
                prompt = first_batch['prompt'][0]
                if 'Answer:' in prompt:
                    answer_pos = prompt.find('Answer:')
                    print(f"\nPrompt ending (showing Answer: marker):")
                    print(prompt[answer_pos-50:answer_pos+200])


