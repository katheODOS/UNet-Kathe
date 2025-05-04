import os
import sys
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path to fix import issues
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def extract_metrics_from_report(report_path):
    """Extract metrics from an evaluation report"""
    metrics = {}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract overall accuracy
            overall_match = re.search(r'Overall Accuracy: ([\d.]+)', content)
            if overall_match:
                metrics['overall_accuracy'] = float(overall_match.group(1))
                
            # Extract per-class metrics
            class_pattern = r'Class (\d+):\nAccuracy: ([\d.]+)\nTotal pixels: (\d+)\nCorrectly classified: (\d+)\nMean per-image accuracy: ([\d.]+)'
            class_matches = re.findall(class_pattern, content)
            
            class_metrics = {}
            for class_id, accuracy, total_pixels, correct_pixels, mean_accuracy in class_matches:
                class_metrics[class_id] = {
                    'accuracy': float(accuracy),
                    'total_pixels': int(total_pixels),
                    'correct_pixels': int(correct_pixels),
                    'mean_accuracy': float(mean_accuracy)
                }
            
            metrics['class_metrics'] = class_metrics
            
        return metrics
    except Exception as e:
        print(f"Error processing {report_path}: {str(e)}")
        return None

def extract_validation_score(output_file):
    """Extract validation Dice score from output file"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'wandb: validation Dice\s+(\d+\.\d+)', content)
            if matches:
                return float(matches[-1])
    except Exception as e:
        print(f"Error processing {output_file}: {str(e)}")
    return None

def extract_dataset_info(model_dir_name):
    """Extract dataset information from model directory name"""
    if 'DSA' in model_dir_name:
        dataset = 'Dataset D SA'
    elif 'ASA' in model_dir_name:
        dataset = 'Dataset A SA'
    elif 'BSA' in model_dir_name:
        dataset = 'Dataset B SA'
    elif 'CSA' in model_dir_name:
        dataset = 'Dataset C SA'
    elif model_dir_name.startswith('A'):
        dataset = 'Dataset A'
    elif model_dir_name.startswith('B'):
        dataset = 'Dataset B'
    elif model_dir_name.startswith('C'):
        dataset = 'Dataset C'
    elif model_dir_name.startswith('D'):
        dataset = 'Dataset D'
  
    # Try to extract learning rate if present
    lr_match = re.search(r'L(\d+e-\d+)', model_dir_name)
    lr = lr_match.group(1) if lr_match else 'Unknown LR'
    
    return dataset, lr

def create_metrics_diff_file(model_dir, overall_accuracy, validation_score):
    """Create a file with difference between overall accuracy and validation dice score"""
    if overall_accuracy is None or validation_score is None:
        return
    
    # Get dataset info and model details
    dataset, lr = extract_dataset_info(model_dir.name)
    
    # Calculate difference
    diff = overall_accuracy - validation_score  # Now using actual difference, not absolute
    abs_diff = abs(diff)
    
    # Create file name that includes model info
    metrics_path = model_dir / f'metrics_comparison_{dataset.replace(" ", "_")}_{lr}.txt'
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Metrics Comparison - {model_dir.name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Learning Rate: {lr}\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.6f}\n")
        f.write(f"Validation Dice F1: {validation_score:.6f}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Difference (Accuracy - F1): {diff:.6f}\n")
        f.write(f"Absolute Difference: {abs_diff:.6f}\n\n")
    
    return diff

def process_directory_recursive(dir_path, results=None):
    """Process a directory recursively to find model results"""
    if results is None:
        results = {}
    
    # Check if this directory contains output.txt - if so, it's a model directory
    output_file = dir_path / 'output.txt'
    results_dir = dir_path / 'results'
    eval_report = results_dir / 'evaluation_report.txt' if results_dir.exists() else None
    
    if output_file.exists() and eval_report is not None and eval_report.exists():
        # This is a valid model directory with evaluation results
        metrics = {}
        
        # Extract metrics from evaluation report
        report_metrics = extract_metrics_from_report(eval_report)
        if report_metrics:
            metrics.update(report_metrics)
        
        # Extract validation score from output.txt
        validation_score = extract_validation_score(output_file)
        if validation_score:
            metrics['validation_score'] = validation_score
            
        # Create metrics diff file if we have both values
        if 'overall_accuracy' in metrics and 'validation_score' in metrics:
            diff = create_metrics_diff_file(
                dir_path, 
                metrics['overall_accuracy'], 
                metrics['validation_score']
            )
            if diff is not None:
                metrics['metrics_difference'] = diff
        
        results[dir_path.name] = metrics
        return results
    
    # If no output.txt or results found, recurse into subdirectories
    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            process_directory_recursive(subdir, results)
    
    return results

def check_class_accuracies(class_metrics):
    """
    Check if all classes (except class 0) have mean per-image accuracy over 0.0
    
    Args:
        class_metrics: Dictionary of class metrics from evaluation report
        
    Returns:
        bool: True if all classes (except 0) have mean_accuracy > 0.0
    """
    if not class_metrics:
        return False
        
    for class_id, metrics in class_metrics.items():
        # Skip class 0 (background/ignore)
        if class_id == '0':
            continue
            
        # Check if mean_accuracy exists and is > 0
        if 'mean_accuracy' not in metrics or metrics['mean_accuracy'] <= 0.0:
            return False
            
    return True

def find_optimal_runs(results):
    """
    Find runs with OA-F1 < 0.1 and all classes with mean accuracy > 0.0
    
    Args:
        results: Dictionary of results for all processed runs
        
    Returns:
        list: List of (model_name, metrics) tuples for optimal runs
    """
    optimal_runs = []
    
    for model_name, metrics in results.items():
        # Skip if missing required metrics
        if not metrics or 'metrics_difference' not in metrics or 'class_metrics' not in metrics:
            continue
            
        # Check if difference is less than 0.1 (using absolute value)
        if abs(metrics['metrics_difference']) < 0.1:
            # Check if all classes have mean accuracy > 0.0
            if check_class_accuracies(metrics['class_metrics']):
                optimal_runs.append((model_name, metrics))
                
    return optimal_runs

def process_checkpoints():
    """Process all checkpoint directories recursively"""
    checkpoints_dir = Path('./checkpoints')
    
    # Process all directories recursively
    results = process_directory_recursive(checkpoints_dir)
    
    # Create overall_accuracies.json
    overall_accuracies = {k: v.get('overall_accuracy', 0) for k, v in results.items() if v and 'overall_accuracy' in v}
    
    with open(checkpoints_dir / 'overall_accuracies.json', 'w', encoding='utf-8') as f:
        json.dump(overall_accuracies, f, indent=4)
    
    # Create validation_scores.json
    validation_scores = {k: v.get('validation_score', 0) for k, v in results.items() if v and 'validation_score' in v}
    
    with open(checkpoints_dir / 'validation_scores.json', 'w', encoding='utf-8') as f:
        json.dump(validation_scores, f, indent=4)
    
    # Create metrics_differences.json
    metric_diffs = {k: v.get('metrics_difference', 0) for k, v in results.items() if v and 'metrics_difference' in v}
    
    with open(checkpoints_dir / 'metrics_differences.json', 'w', encoding='utf-8') as f:
        json.dump(metric_diffs, f, indent=4)
    
    # Sort and create sorted text files
    sorted_accuracies = sorted(overall_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    with open(checkpoints_dir / 'overall_accuracies_sorted.txt', 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Results (Sorted by Overall Accuracy)\n")
        f.write("="*60 + "\n\n")
        
        for i, (model_name, accuracy) in enumerate(sorted_accuracies, 1):
            validation = results[model_name].get('validation_score', 0)
            diff = results[model_name].get('metrics_difference', 'N/A')
            diff_str = f"{diff:.4f}" if isinstance(diff, float) else diff
            f.write(f"{i}. {model_name}: Acc={accuracy:.4f}, Val={validation:.4f}, Diff={diff_str}\n")
    
    # Generate visualizations for top performers
    generate_visualizations(results, checkpoints_dir)
    
    # Find optimal runs based on criteria
    optimal_runs = find_optimal_runs(results)
    
    # Sort optimal runs by overall accuracy
    optimal_runs.sort(key=lambda x: x[1].get('overall_accuracy', 0), reverse=True)
    
    # Save optimal runs to optimal_runs.txt
    with open(checkpoints_dir / 'optimal_runs.txt', 'w', encoding='utf-8') as f:
        f.write("Optimal Model Runs\n")
        f.write("=================\n\n")
        f.write("Criteria:\n")
        f.write("1. |Overall Accuracy - F1 Score| < 0.1\n")
        f.write("2. All classes (except 0) have mean per-image accuracy > 0.0\n\n")
        
        if not optimal_runs:
            f.write("No models match the criteria.\n")
        else:
            f.write(f"Found {len(optimal_runs)} optimal models:\n\n")
            
            for i, (model_name, metrics) in enumerate(optimal_runs, 1):
                dataset, lr = extract_dataset_info(model_name)
                oa = metrics.get('overall_accuracy', 0)
                f1 = metrics.get('validation_score', 0)
                diff = metrics.get('metrics_difference', 0)
                
                f.write(f"{i}. {model_name}\n")
                f.write(f"   Dataset: {dataset}\n")
                f.write(f"   Learning Rate: {lr}\n")
                f.write(f"   Overall Accuracy: {oa:.4f}\n")
                f.write(f"   F1 Score: {f1:.4f}\n")
                f.write(f"   Difference: {diff:.4f}\n\n")
    
    # Print optimal runs to terminal
    print("\n" + "="*60)
    print("OPTIMAL MODEL RUNS")
    print("="*60)
    print("Models with |OA-F1| < 0.1 and all classes having mean accuracy > 0.0:")
    print("-"*60)
    
    if not optimal_runs:
        print("No models match the criteria.")
    else:
        for i, (model_name, metrics) in enumerate(optimal_runs, 1):
            oa = metrics.get('overall_accuracy', 0)
            f1 = metrics.get('validation_score', 0)
            diff = metrics.get('metrics_difference', 0)
            print(f"{i}. {model_name}")
            print(f"   OA: {oa:.4f}, F1: {f1:.4f}, Diff: {diff:.4f}")
    
    print("="*60)
    print(f"Full details saved to {checkpoints_dir / 'optimal_runs.txt'}")
    
    print(f"Processed {len(results)} model directories")
    print(f"Created JSON files and sorted rankings")
    
    # Print top 5 models
    print("\nTop 5 Models by Overall Accuracy:")
    for i, (model_name, accuracy) in enumerate(sorted_accuracies[:5], 1):
        validation = results[model_name].get('validation_score', 0)
        diff = results[model_name].get('metrics_difference', 'N/A')
        diff_str = f"{diff:.4f}" if isinstance(diff, float) else diff
        print(f"{i}. {model_name}: Acc={accuracy:.4f}, Val={validation:.4f}, Diff={diff_str}")

def process_checkpoint_dir(model_dir):
    """Process a single checkpoint directory and extract metrics"""
    results = {}
    
    # Check for evaluation report
    eval_report = model_dir / 'results' / 'evaluation_report.txt'
    if eval_report.exists():
        metrics = extract_metrics_from_report(eval_report)
        if metrics:
            results.update(metrics)
    
    # Check for output.txt for validation scores
    output_file = model_dir / 'output.txt'
    if output_file.exists():
        validation_score = extract_validation_score(output_file)
        if validation_score:
            results['validation_score'] = validation_score
    
    return results

def generate_visualizations(results, output_dir):
    """Generate visualizations comparing top performers"""
    # Get top 10 models by overall accuracy
    top_models = sorted(
        [(k, v) for k, v in results.items() if v and 'overall_accuracy' in v],
        key=lambda x: x[1]['overall_accuracy'],
        reverse=True
    )[:10]
    
    if not top_models:
        return
    
    # Create visualization directory
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Plot overall accuracies
    plt.figure(figsize=(15, 8))
    models = [m[0] for m in top_models]
    accuracies = [m[1]['overall_accuracy'] for m in top_models]
    
    # Use YlGn colormap from matplotlib
    cmap = plt.get_cmap('YlGn')
    colors = [cmap(i) for i in np.linspace(0.3, 0.9, len(models))]
    
    bars = plt.bar(range(len(models)), accuracies, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)
    
    plt.title('Top Models by Overall Accuracy', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Overall Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(range(len(models)), [m[:10] + '...' if len(m) > 10 else m for m in models], rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(vis_dir / 'top_models_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_dataset_rankings(dataset_name, models, output_dir):
    """Generate rankings for models within a specific dataset"""
    if not models:
        return
    
    # Create dataset visualizations directory
    dataset_dir = output_dir / 'dataset_comparisons'
    dataset_dir.mkdir(exist_ok=True)
    
    # Filter for models with overall accuracy
    models_with_metrics = [(name, metrics) for name, metrics in models if metrics and 'overall_accuracy' in metrics]
    
    # Sort by overall accuracy
    sorted_models = sorted(models_with_metrics, key=lambda x: x[1]['overall_accuracy'], reverse=True)
    
    # Generate report
    with open(dataset_dir / f'{dataset_name}_rankings.txt', 'w', encoding='utf-8') as f:
        f.write(f"Models for {dataset_name} (Sorted by Overall Accuracy)\n")
        f.write("="*60 + "\n\n")
        
        for i, (model_name, metrics) in enumerate(sorted_models, 1):
            accuracy = metrics.get('overall_accuracy', 0)
            validation = metrics.get('validation_score', 0)
            f.write(f"{i}. {model_name}: Overall Acc={accuracy:.4f}, Validation={validation:.4f}\n")

if __name__ == '__main__':
    process_checkpoints()
