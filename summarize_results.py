#!/usr/bin/env python3
"""
Script to summarize all experiment results and display accuracy in a table format.
This script reads the extended JSON files and creates a summary table.
"""

import os
import json
import pandas as pd
from pathlib import Path

def find_result_files(base_dir="data/gen_res", llm_id="meta-llama/Llama-2-7b-chat-hf", split="train"):
    """Find all extended result JSON files."""
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    results_dir = Path(base_dir) / llm_folder / split / "classic" / "contriever" / "7_doc"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []
    
    # Find all extended JSON files
    result_files = list(results_dir.glob("*all_extended.json"))
    return result_files

def extract_config_from_filename(filename):
    """Extract configuration from filename."""
    name = filename.stem.replace("_all_extended", "")
    
    # Parse: numdoc7_gold_at{pos}_{rand?}_{answerless?}_info
    config = {
        "gold_position": None,
        "use_random": False,
        "scenario": None
    }
    
    if "gold_at0" in name:
        config["gold_position"] = 0
        config["scenario"] = "far"
    elif "gold_at3" in name:
        config["gold_position"] = 3
        config["scenario"] = "mid"
    elif "gold_at6" in name:
        config["gold_position"] = 6
        config["scenario"] = "near"
    
    if "_rand" in name:
        config["use_random"] = True
    
    return config

def calculate_accuracy(result_file):
    """Calculate accuracy from result file."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        if 'ans_match_after_norm' not in df.columns:
            return None
        
        accuracy = df['ans_match_after_norm'].sum() / len(df)
        return round(accuracy, 4)
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return None

def main():
    result_files = find_result_files()
    
    if not result_files:
        print("No result files found. Make sure you have run the evaluation script first.")
        print("Run: python src/read_generation_results.py with appropriate parameters")
        return
    
    results = []
    
    for result_file in result_files:
        config = extract_config_from_filename(result_file)
        accuracy = calculate_accuracy(result_file)
        
        if accuracy is not None and config["scenario"]:
            results.append({
                "Scenario": config["scenario"],
                "Document Type": "Random" if config["use_random"] else "Relevant",
                "Gold Position": config["gold_position"],
                "Accuracy": accuracy,
                "File": result_file.name
            })
    
    if not results:
        print("No valid results found.")
        return
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values(["Document Type", "Scenario"])
    
    # Display results
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Table 1: Random Documents
    print("Table 1: Random Documents")
    print("-" * 80)
    random_df = df[df["Document Type"] == "Random"]
    if not random_df.empty:
        for _, row in random_df.iterrows():
            print(f"  {row['Scenario'].upper():6s} (gold at {row['Gold Position']}): {row['Accuracy']:.4f}")
    else:
        print("  No results available")
    print()
    
    # Table 2: Relevant Documents
    print("Table 2: Relevant/Distracting Documents")
    print("-" * 80)
    relevant_df = df[df["Document Type"] == "Relevant"]
    if not relevant_df.empty:
        for _, row in relevant_df.iterrows():
            print(f"  {row['Scenario'].upper():6s} (gold at {row['Gold Position']}): {row['Accuracy']:.4f}")
    else:
        print("  No results available")
    print()
    
    # Summary table
    print("Summary Table (for easy copy-paste)")
    print("-" * 80)
    print(df.to_string(index=False))
    print()
    
    # Save to CSV
    output_file = "results_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()



