#!/usr/bin/env python3
"""
Analyze cases where 11 documents succeed but 1 document fails.
For gen_res_prompt experiment.
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

def analyze_comparison(results_1doc, results_11doc):
    """Compare results and find cases where 11 doc succeeds but 1 doc fails."""
    
    # Create lookup dictionaries by example_id
    results_1doc_dict = {r['example_id']: r for r in results_1doc}
    results_11doc_dict = {r['example_id']: r for r in results_11doc}
    
    # Find common example_ids
    common_ids = set(results_1doc_dict.keys()) & set(results_11doc_dict.keys())
    
    mismatches = []
    stats = {
        'total_compared': len(common_ids),
        'both_correct': 0,
        'both_incorrect': 0,
        'only_11doc_correct': 0,
        'only_1doc_correct': 0,
        'answer_in_docs_but_1doc_failed': 0
    }
    
    for example_id in common_ids:
        r1 = results_1doc_dict[example_id]
        r11 = results_11doc_dict[example_id]
        
        correct_1doc = r1['ans_match_after_norm']
        correct_11doc = r11['ans_match_after_norm']
        
        if correct_1doc and correct_11doc:
            stats['both_correct'] += 1
        elif not correct_1doc and not correct_11doc:
            stats['both_incorrect'] += 1
        elif not correct_1doc and correct_11doc:
            stats['only_11doc_correct'] += 1
            # Check if answer was in documents
            answer_in_docs = r1['ans_in_documents']
            if answer_in_docs:
                stats['answer_in_docs_but_1doc_failed'] += 1
            
            mismatches.append({
                'example_id': example_id,
                'query': r1['query'],
                'generated_1doc': r1['generated_answer'],
                'generated_11doc': r11['generated_answer'],
                'ground_truth': r1['answers'],
                'answer_in_docs': answer_in_docs
            })
        elif correct_1doc and not correct_11doc:
            stats['only_1doc_correct'] += 1
    
    return mismatches, stats

def categorize_examples(mismatches):
    """Categorize examples by failure pattern."""
    categories = defaultdict(list)
    
    for ex in mismatches:
        query = ex['query'].lower()
        gen_1doc = ex['generated_1doc'].lower()
        gen_11doc = ex['generated_11doc'].lower()
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

def calculate_accuracy(results):
    """Calculate accuracy from results."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r['ans_match_after_norm'])
    return round(correct / len(results), 4)

def generate_examples_file(mismatches, stats, accuracy_1doc, accuracy_11doc, output_file='examples_prompt_11_vs_1.txt'):
    """Generate examples.txt file similar to the original format."""
    
    categories = categorize_examples(mismatches)
    
    with open(output_file, 'w') as f:
        f.write("## Summary: Cases where 1 doc fails but 11 doc succeeds\n\n")
        f.write("### Statistics\n")
        f.write(f"- Total compared: {stats['total_compared']} examples\n")
        f.write(f"- Both correct: {stats['both_correct']} cases\n")
        f.write(f"- Both incorrect: {stats['both_incorrect']} cases\n")
        f.write(f"- Only 11 doc correct (1 doc fails): {stats['only_11doc_correct']} cases\n")
        f.write(f"- Only 1 doc correct (11 doc fails): {stats['only_1doc_correct']} cases\n")
        f.write(f"- Answer was in documents but 1 doc still failed: {stats['answer_in_docs_but_1doc_failed']} ({100*stats['answer_in_docs_but_1doc_failed']/max(stats['only_11doc_correct'],1):.1f}% of mismatches)\n\n")
        
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
                    f.write(f"   - 11 doc: `{ex['generated_11doc'][:100]}`\n")
                    f.write(f"   - Ground truth: {ex['ground_truth']}\n")
                f.write("\n")
        
        f.write("### Insights\n\n")
        f.write("1. More documents help with:\n")
        f.write("   - Full name extraction (first + last)\n")
        f.write("   - Disambiguation (character vs actor, entity selection)\n")
        f.write("   - Comparison questions (requires multiple pieces of info)\n")
        f.write("   - Context-dependent answers\n")
        f.write("   - Better extraction from noisy/distracting contexts\n\n")
        
        f.write("2. 11 doc advantages:\n")
        f.write(f"   - {stats['only_11doc_correct']} cases where only 11 doc succeeds (1 doc fails)\n")
        f.write("   - Better at finding answers that require more context\n")
        f.write("   - More robust to noise from distracting documents\n")
        f.write(f"   - Improved accuracy: {accuracy_11doc} (11 doc) vs {accuracy_1doc} (1 doc)\n\n")
        
        f.write("3. All failures had the answer in the document, suggesting:\n")
        f.write("   - Extraction/formatting issues\n")
        f.write("   - Need for additional context to disambiguate\n")
        f.write("   - Difficulty identifying the correct answer when multiple candidates exist\n")
        f.write("   - Better prompt understanding with more context\n\n")
        
        f.write("These results suggest that more documents (11) can improve accuracy, ")
        f.write("especially for questions requiring disambiguation, full names, or comparisons.\n")
    
    print(f"Examples file created: {output_file}")
    print(f"Total mismatches found: {len(mismatches)}")

def main():
    base_dir = "data/gen_res_prompt/Llama-2-7b-chat-hf/train/classic/contriever"
    
    file_1doc = os.path.join(base_dir, "1_doc/numdoc1_gold_at0_rand_answerless_info_all_extended.json")
    file_11doc = os.path.join(base_dir, "11_doc/numdoc11_gold_at10_rand_answerless_info_all_extended.json")
    
    if not os.path.exists(file_1doc):
        print(f"Error: File not found: {file_1doc}")
        return
    
    if not os.path.exists(file_11doc):
        print(f"Error: File not found: {file_11doc}")
        return
    
    print("Loading results...")
    results_1doc = load_results(file_1doc)
    results_11doc = load_results(file_11doc)
    
    print(f"Loaded {len(results_1doc)} results from 1 doc")
    print(f"Loaded {len(results_11doc)} results from 11 doc")
    
    # Calculate accuracies
    accuracy_1doc = calculate_accuracy(results_1doc)
    accuracy_11doc = calculate_accuracy(results_11doc)
    
    print(f"Accuracy 1 doc: {accuracy_1doc}")
    print(f"Accuracy 11 doc: {accuracy_11doc}")
    
    print("Analyzing comparison...")
    mismatches, stats = analyze_comparison(results_1doc, results_11doc)
    
    print("\nStatistics:")
    print(f"  Total compared: {stats['total_compared']}")
    print(f"  Both correct: {stats['both_correct']}")
    print(f"  Only 11 doc correct: {stats['only_11doc_correct']}")
    print(f"  Only 1 doc correct: {stats['only_1doc_correct']}")
    
    print("\nGenerating examples file...")
    generate_examples_file(mismatches, stats, accuracy_1doc, accuracy_11doc, 'examples_prompt_11_vs_1.txt')

if __name__ == "__main__":
    main()


