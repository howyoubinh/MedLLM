# Adapted from /SFT/eval/eval_medqa.py
import os
import argparse
from typing import Sequence, Dict, Tuple
from tqdm import tqdm
import re
import json
import jsonlines
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, help='Path to the JSONL result file')
    parser.add_argument('--output-dir', type=str, help='Directory to save evaluation results')
    args = parser.parse_args()
    return args

def extract_predicted_answer(pmc_output: str) -> str:
    """Extract predicted answer from model output"""
    matched_pieces = re.findall(r'(?i)OPTION [ABCD] IS CORRECT', pmc_output)
    if len(matched_pieces) == 0:
        return "NONE"
    predicted_option = matched_pieces[0].split()[1]
    return predicted_option

def evaluate_predictions(results: list) -> Tuple[Dict, np.ndarray, str]:
    """Calculate evaluation metrics"""
    y_true = []
    y_pred = []
    no_answer = 0
    no_answer_indicies = []
    
    for idx, item in enumerate(results):
        true_answer = item['Answer']
        pred_answer = extract_predicted_answer(item['pmc_output'])
        
        if pred_answer == "NONE":
            no_answer += 1
            no_answer_indicies.append(idx + 1)
            continue
            
        y_true.append(true_answer)
        y_pred.append(pred_answer)

    # Calculate metrics
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0

    # Create confusion matrix
    labels = ['A', 'B', 'C', 'D']
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred, labels=labels)

    metrics = {
        'total_samples': len(results),
        'no_answer': no_answer,
        'no_answer_indicies': no_answer_indicies,
        'valid_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
    }

    return metrics, conf_matrix, class_report, no_answer_indicies

def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: str):
    """Plot and save confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                xticklabels=['A', 'B', 'C', 'D'],
                yticklabels=['A', 'B', 'C', 'D'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = []
    with jsonlines.open(args.result_file) as reader:
        for obj in reader:
            results.append(obj)

    # Calculate metrics
    metrics, conf_matrix, class_report, no_answer_indices = evaluate_predictions(results)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Append no-answer details to classification report
    class_report = (f"{class_report}\n\n"
                      f"No Answer Details:\n"
                      f"------------------\n"
                      f"Total no-answer samples: {metrics['no_answer']}\n"
                      f"JSONL line numbers with no answer: {no_answer_indices}")
        
    # Save classification report
    report_file = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(class_report)
        
    # Plot and save confusion matrix
    plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, plot_path)
    
    # Print results to console
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"No answer: {metrics['no_answer']}")
    print(f"No answer indices: {no_answer_indices}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("\nClassification Report:")
    print(class_report)

if __name__ == '__main__':
    main()