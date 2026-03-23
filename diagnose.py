#!/usr/bin/env python3

"""
T型流体场预测质量诊断工具
分析PDF输出，评估预测精度，提出改进建议
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_header(title):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{title.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}\n")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.END} {msg}")

def print_metric(name, value, unit="", threshold_good=None, threshold_ok=None):
    """打印指标，带颜色标记"""
    if threshold_good is not None and threshold_ok is not None:
        if value >= threshold_good:
            color = Colors.GREEN
            status = "✓ Good"
        elif value >= threshold_ok:
            color = Colors.YELLOW
            status = "~ OK"
        else:
            color = Colors.RED
            status = "✗ Poor"
        print(f"  {name:25s}: {color}{value:10.6f}{Colors.END} {unit:10s} [{status}]")
    else:
        print(f"  {name:25s}: {value:10.6f} {unit}")

def load_training_logs(run_dir):
    """加载训练日志"""
    logs = {}
    
    # 尝试加载JSON日志
    log_file = Path(run_dir) / 'args.txt'
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            print_success(f"Loaded logs from {log_file}")
        except Exception as e:
            print_warning(f"Failed to load {log_file}: {e}")
    
    return logs

def load_metrics(train_dir):
    """加载训练指标"""
    metrics = {}
    
    # 加载R²指标
    r2_file = Path(train_dir) / 'r2.txt'
    if r2_file.exists():
        try:
            r2_test = np.loadtxt(r2_file)
            metrics['r2_test'] = r2_test
            metrics['final_r2'] = float(r2_test[-1]) if len(r2_test) > 0 else 0
        except:
            pass
    
    # 加载RMSE指标
    rmse_file = Path(train_dir) / 'rmse_test.txt'
    if rmse_file.exists():
        try:
            rmse_test = np.loadtxt(rmse_file)
            metrics['rmse_test'] = rmse_test
            metrics['final_rmse'] = float(rmse_test[-1]) if len(rmse_test) > 0 else 0
        except:
            pass
    
    return metrics

def analyze_predictions(pred_dir):
    """分析预测文件"""
    analysis = {
        'total_predictions': 0,
        'epochs_with_predictions': set(),
    }
    
    if not Path(pred_dir).exists():
        print_warning(f"Prediction directory not found: {pred_dir}")
        return analysis
    
    # 统计预测文件
    pred_files = list(Path(pred_dir).glob('*.pdf'))
    analysis['total_predictions'] = len(pred_files)
    
    for pdf_file in pred_files:
        # 尝试从文件名提取epoch信息
        try:
            name_parts = pdf_file.stem.split('_')
            if name_parts[0] == 'pred' and 'epoch' in name_parts[1]:
                epoch_str = name_parts[1].replace('epoch', '')
                analysis['epochs_with_predictions'].add(int(epoch_str))
        except:
            pass
    
    if analysis['epochs_with_predictions']:
        analysis['latest_epoch'] = max(analysis['epochs_with_predictions'])
        analysis['earliest_epoch'] = min(analysis['epochs_with_predictions'])
    
    return analysis

def diagnose_model_quality(metrics):
    """诊断模型质量并给出建议"""
    print_header("Model Quality Diagnosis")
    
    issues = []
    suggestions = []
    
    if 'final_r2' not in metrics or 'final_rmse' not in metrics:
        print_warning("Insufficient metrics data for diagnosis")
        return
    
    r2 = metrics.get('final_r2', 0)
    rmse = metrics.get('final_rmse', 0)
    
    print("Quality Metrics:")
    print_metric("R² Score", r2, threshold_good=0.95, threshold_ok=0.90)
    print_metric("RMSE", rmse, threshold_good=0.03, threshold_ok=0.05)
    
    print("\nDiagnosis:")
    
    # R²诊断
    if r2 < 0.80:
        issues.append("R² score is too low (< 0.80)")
        suggestions.append("• Increase model capacity: --growth-rate 24-32")
        suggestions.append("• Increase training epochs to 400+")
        suggestions.append("• Try 'advanced' optimization plan")
    elif r2 < 0.90:
        issues.append("R² score could be better (< 0.90)")
        suggestions.append("• Run more epochs (300-400)")
        suggestions.append("• Fine-tune learning rate")
    elif r2 < 0.95:
        issues.append("R² score is good but not optimal (< 0.95)")
        suggestions.append("• Consider 'advanced' plan with more epochs")
    else:
        print_success("✓ R² score is excellent (≥ 0.95)!")
    
    # RMSE诊断
    if rmse > 0.08:
        issues.append("RMSE is high (> 0.08)")
        suggestions.append("• Check data normalization")
        suggestions.append("• Verify T-mask configuration")
    elif rmse > 0.05:
        issues.append("RMSE could be improved (> 0.05)")
        suggestions.append("• Increase gradient loss weight")
    elif rmse > 0.03:
        print_success("✓ RMSE is good (< 0.05)")
    else:
        print_success("✓ RMSE is excellent (< 0.03)!")
    
    # 收敛性诊断
    if 'r2_test' in metrics and len(metrics['r2_test']) > 1:
        r2_array = metrics['r2_test']
        improvement_rate = (r2_array[-1] - r2_array[0]) / max(abs(r2_array[0]), 1e-6)
        
        if improvement_rate < 0.05:
            issues.append("Slow convergence")
            suggestions.append("• Try lower learning rate or different schedule")
        elif improvement_rate < 0.15:
            issues.append("Convergence could be faster")
            suggestions.append("• Increase batch size")
        else:
            print_success("✓ Good convergence rate")
    
    # 输出问题和建议
    if issues:
        print("\nIdentified Issues:")
        for issue in issues:
            print_error(f"• {issue}")
    else:
        print_success("✓ No major issues detected!")
    
    if suggestions:
        print("\nRecommendations:")
        for suggestion in suggestions:
            print_info(suggestion)

def find_latest_experiment():
    """查找最新的实验目录"""
    exp_dir = Path('experiments/deterministic/kle4225')
    
    if not exp_dir.exists():
        return None
    
    # 查找所有以ntrain开头的目录
    exp_dirs = sorted(exp_dir.glob('ntrain*'))
    
    if not exp_dirs:
        return None
    
    # 返回最新的（最后一个）
    return exp_dirs[-1]

def main():
    print_header("T-Channel Fluid Field Prediction Quality Diagnostic")
    
    # 查找实验目录
    exp_dir = find_latest_experiment()
    
    if not exp_dir:
        print_error("No experiment directory found!")
        print_info("Expected: experiments/deterministic/kle4225/ntrain*")
        return 1
    
    print_info(f"Found experiment: {exp_dir.name}")
    print_info(f"Full path: {exp_dir}")
    
    # 定义路径
    run_dir = exp_dir
    train_dir = exp_dir / 'training'
    pred_dir = train_dir / 'predictions'
    
    # 加载日志和指标
    logs = load_training_logs(run_dir)
    metrics = load_metrics(train_dir)
    pred_analysis = analyze_predictions(pred_dir)
    
    # 显示实验配置
    print_header("Experiment Configuration")
    if logs:
        important_keys = ['ntrain', 'ntest', 'epochs', 'batch_size', 'lr', 
                         'weight_decay', 'blocks', 'growth_rate']
        for key in important_keys:
            if key in logs:
                print(f"  {key:20s}: {logs[key]}")
    
    # 显示训练结果
    print_header("Training Results")
    if metrics:
        if 'final_r2' in metrics:
            print_metric("Final R²", metrics['final_r2'], threshold_good=0.95, threshold_ok=0.90)
        if 'final_rmse' in metrics:
            print_metric("Final RMSE", metrics['final_rmse'], threshold_good=0.03, threshold_ok=0.05)
    
    # 显示预测统计
    print_header("Prediction Files")
    print_info(f"Total PDF files: {pred_analysis['total_predictions']}")
    if pred_analysis['epochs_with_predictions']:
        print_info(f"Epochs with predictions: {pred_analysis['earliest_epoch']} - {pred_analysis['latest_epoch']}")
    
    if pred_dir.exists():
        latest_pdf = sorted(pred_dir.glob('*.pdf'))
        if latest_pdf:
            print_success(f"\nLatest prediction file: {latest_pdf[-1].name}")
            print_info(f"View with: open {latest_pdf[-1]} (or double-click the file)")
    
    # 诊断
    if metrics:
        diagnose_model_quality(metrics)
    
    # 显示后续步骤
    print_header("Next Steps")
    print_info("1. Open the latest PDF file to visually inspect predictions")
    print_info("2. Compare:")
    print_info("   - Row 1: Target image")
    print_info("   - Row 2: Predicted image (should be very similar to row 1)")
    print_info("   - Row 3: Error map (should have small values, displayed as darker)")
    print_info("")
    print_info("3. If quality is not satisfactory, try:")
    print_info("   python run_training.py quick    # Quick optimization")
    print_info("   python run_training.py advanced # Advanced optimization")
    print_info("")
    print_info("4. View training curves:")
    print_info(f"   {train_dir}/r2.pdf - R² score evolution")
    print_info(f"   {train_dir}/rmse.pdf - RMSE evolution")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
