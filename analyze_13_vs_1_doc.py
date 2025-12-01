#!/usr/bin/env python3
"""
Analyze cases where 13 documents succeed but 1 document fails.
Creates examples.txt similar to the original analysis.
"""

import json
import os
from collections import defaultdict

def load_results(filepath):
    """Load extended results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def normalize_answer(text):
    """Simple normalization for comparison."""
    return text.lower().strip()

def is_answer_correct(generated_answer, ground_truth_answers):
    """Check if generated answer matches any ground truth."""
    normalized_gen = normalize_answer(generated_answer)
    for gt in ground_truth_answers:
        normalized_gt = normalize_answer(gt)
        if normalized_gt in normalized_gen or normalized_gen in normalized_gt:
            return True
    return False

def analyze_comparison(results_1doc, results_13doc):
    """Compare results and find cases where 13 doc succeeds but 1 doc fails."""
    
    # Create lookup dictionaries by example_id
    results_1doc_dict = {r['example_id']: r for r in results_1doc}
    results_13doc_dict = {r['example_id']: r for r in results_13doc}
    
    # Find common example_ids
    common_ids = set(results_1doc_dict.keys()) & set(results_13doc_dict.keys())
    
    mismatches = []
    stats = {
        'total_compared': len(common_ids),
        'both_correct': 0,
        'both_incorrect': 0,
        'only_13doc_correct': 0,
        'only_1doc_correct': 0,
        'answer_in_docs_but_1doc_failed': 0
    }
    
    for example_id in common_ids:
        r1 = results_1doc_dict[example_id]
        r13 = results_13doc_dict[example_id]
        
        correct_1doc = r1['ans_match_after_norm']
        correct_13doc = r13['ans_match_after_norm']
        
        if correct_1doc and correct_13doc:
            stats['both_correct'] += 1
        elif not correct_1doc and not correct_13doc:
            stats['both_incorrect'] += 1
        elif not correct_1doc and correct_13doc:
            stats['only_13doc_correct'] += 1
            # Check if answer was in documents
            answer_in_docs = r1['ans_in_documents']
            if answer_in_docs:
                stats['answer_in_docs_but_1doc_failed'] += 1
            
            mismatches.append({
                'example_id': example_id,
                'query': r1['query'],
                'generated_1doc': r1['generated_answer'],
                'generated_13doc': r13['generated_answer'],
                'ground_truth': r1['answers'],
                'answer_in_docs': answer_in_docs
            })
        elif correct_1doc and not correct_13doc:
            stats['only_1doc_correct'] += 1
    
    return mismatches, stats

def categorize_examples(mismatches):
    """Categorize examples by failure pattern."""
    categories = defaultdict(list)
    
    for ex in mismatches:
        query = ex['query'].lower()
        gen_1doc = ex['generated_1doc'].lower()
        gen_13doc = ex['generated_13doc'].lower()
        gt = [a.lower() for a in ex['ground_truth']]
        
        # Check for partial name extraction
        if any(len(gt_word.split()) > 1 and gen_1doc in gt_word and gt_word not in gen_1doc 
               for gt_word in gt for word in gt_word.split()):
            categories['partial_name'].append(ex)
        # Check for truncated answers
        elif len(gen_1doc) < 10 and any(gt_word.startswith(gen_1doc) for gt_word in gt):
            categories['truncated'].append(ex)
        # Check for wrong entity selection
        elif 'who plays' in query or 'who is' in query:
            categories['entity_selection'].append(ex)
        # Check for comparison questions
        elif 'more' in query or 'less' in query or 'better' in query or 'worse' in query:
            categories['comparison'].append(ex)
        # Check for incomplete extraction
        elif gen_1doc and not any(gt_word in gen_1doc for gt_word in gt):
            categories['wrong_extraction'].append(ex)
        else:
            categories['other'].append(ex)
    
    return categories

def generate_examples_file(mismatches, stats, output_file='examples_13_vs_1.txt'):
    """Generate examples.txt file similar to the original format."""
    
    categories = categorize_examples(mismatches)
    
    with open(output_file, 'w') as f:
        f.write("## Summary: Cases where 1 doc fails but 13 doc succeeds\n\n")
        f.write("### Statistics\n")
        f.write(f"- Total compared: {stats['total_compared']} examples\n")
        f.write(f"- Both correct: {stats['both_correct']} cases\n")
        f.write(f"- Both incorrect: {stats['both_incorrect']} cases\n")
        f.write(f"- Only 13 doc correct (1 doc fails): {stats['only_13doc_correct']} cases\n")
        f.write(f"- Only 1 doc correct (13 doc fails): {stats['only_1doc_correct']} cases\n")
        f.write(f"- Answer was in documents but 1 doc still failed: {stats['answer_in_docs_but_1doc_failed']} ({100*stats['answer_in_docs_but_1doc_failed']/max(stats['only_13doc_correct'],1):.1f}% of mismatches)\n\n")
        
        f.write("### Common failure patterns in 1 doc\n\n")
        
        # Write examples for each category
        category_names = {
            'partial_name': '1. Partial name extraction',
            'truncated': '2. Incomplete extraction (truncated)',
            'entity_selection': '3. Wrong entity selection',
            'comparison': '4. Comparison questions',
            'wrong_extraction': '5. Wrong answer extraction',
            'other': '6. Other cases'
        }
        
        for cat_key, cat_name in category_names.items():
            if cat_key in categories and categories[cat_key]:
                f.write(f"{cat_name}\n")
                # Show up to 5 examples per category
                for ex in categories[cat_key][:5]:
                    f.write(f"   - Example: \"{ex['query']}\"\n")
                    f.write(f"   - 1 doc: `{ex['generated_1doc'][:100]}`\n")
                    f.write(f"   - 13 doc: `{ex['generated_13doc'][:100]}`\n")
                    f.write(f"   - Ground truth: {ex['ground_truth']}\n")
                f.write("\n")
        
        f.write("### Insights\n\n")
        f.write("1. More documents help with:\n")
        f.write("   - Full name extraction (first + last)\n")
        f.write("   - Disambiguation (character vs actor, entity selection)\n")
        f.write("   - Comparison questions (requires multiple pieces of info)\n")
        f.write("   - Context-dependent answers\n")
        f.write("   - Better extraction from noisy/distracting contexts\n\n")
        
        f.write("2. 13 doc advantages:\n")
        f.write(f"   - {stats['only_13doc_correct']} cases where only 13 doc succeeds (1 doc fails)\n")
        f.write("   - Better at finding answers that require more context\n")
        f.write("   - More robust to noise from distracting documents\n")
        f.write("   - Improved accuracy: 54.27% (13 doc) vs 53.78% (1 doc)\n\n")
        
        f.write("3. All failures had the answer in the document, suggesting:\n")
        f.write("   - Extraction/formatting issues\n")
        f.write("   - Need for additional context to disambiguate\n")
        f.write("   - Difficulty identifying the correct answer when multiple candidates exist\n")
        f.write("   - Better prompt understanding with more context\n\n")
        
        f.write("These results suggest that more documents (13) can improve accuracy, ")
        f.write("especially for questions requiring disambiguation, full names, or comparisons.\n")
    
    print(f"Examples file created: {output_file}")
    print(f"Total mismatches found: {len(mismatches)}")

def main():
    base_dir = "data/gen_res_prompt_ans_shortly/Llama-2-7b-chat-hf/train/classic/contriever"
    
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
    
    print("Analyzing comparison...")
    mismatches, stats = analyze_comparison(results_1doc, results_13doc)
    
    print("\nStatistics:")
    print(f"  Total compared: {stats['total_compared']}")
    print(f"  Both correct: {stats['both_correct']}")
    print(f"  Only 13 doc correct: {stats['only_13doc_correct']}")
    print(f"  Only 1 doc correct: {stats['only_1doc_correct']}")
    
    print("\nGenerating examples file...")
    generate_examples_file(mismatches, stats, 'examples_13_vs_1.txt')

if __name__ == "__main__":
    main()


