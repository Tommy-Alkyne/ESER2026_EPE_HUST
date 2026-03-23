#!/usr/bin/env python3

"""
T型流体场预测模型优化训练启动脚本
支持多种优化方案和参数配置
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# 颜色定义
class Colors:
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    END = '\033[0m'

def print_info(msg):
    print(f"{Colors.INFO}[INFO]{Colors.END} {msg}")

def print_success(msg):
    print(f"{Colors.SUCCESS}[SUCCESS]{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.WARNING}[WARNING]{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.ERROR}[ERROR]{Colors.END} {msg}")

def print_header(title):
    print(f"\n{Colors.INFO}{'='*70}{Colors.END}")
    print(f"{Colors.INFO}{title.center(70)}{Colors.END}")
    print(f"{Colors.INFO}{'='*70}{Colors.END}\n")

# 训练方案定义
PLANS = {
    'baseline': {
        'description': '原始参数对比（用于baseline）',
        'epochs': 200,
        'batch_size': 16,
        'lr': 3e-3,
        'weight_decay': 5e-4,
        'growth_rate': 16,
        'recommended': False,
    },
    'quick': {
        'description': '快速优化方案（新手推荐） ⭐',
        'epochs': 300,
        'batch_size': 32,
        'lr': 2e-3,
        'weight_decay': 8e-4,
        'growth_rate': 16,
        'recommended': True,
    },
    'advanced': {
        'description': '高级优化方案（最佳效果） ⭐⭐',
        'epochs': 400,
        'batch_size': 64,
        'lr': 1.5e-3,
        'weight_decay': 1e-3,
        'growth_rate': 24,
        'recommended': False,
    },
    'experimental': {
        'description': '实验方案（激进优化）',
        'epochs': 500,
        'batch_size': 128,
        'lr': 1e-3,
        'weight_decay': 1.2e-3,
        'growth_rate': 32,
        'recommended': False,
    },
}

def show_plans():
    """显示所有可用的训练方案"""
    print_header("Available Training Plans")
    
    for plan_name, config in PLANS.items():
        mark = "  👉 " if config['recommended'] else "     "
        print(f"{mark}Plan: {plan_name}")
        print(f"       {config['description']}")
        print(f"       - Epochs: {config['epochs']}")
        print(f"       - Batch Size: {config['batch_size']}")
        print(f"       - Learning Rate: {config['lr']}")
        print(f"       - Weight Decay: {config['weight_decay']}")
        print(f"       - Growth Rate: {config['growth_rate']}")
        print()

def validate_environment():
    """检查运行环境"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_info(f"Working directory: {script_dir}")
    
    # 检查train_det.py
    if not Path('train_det.py').exists():
        print_error("train_det.py not found!")
        return False
    
    # 检查数据集
    if not Path('dataset/p_v_256.hdf5').exists():
        print_error("dataset/p_v_256.hdf5 not found!")
        return False
    
    print_success("Environment check passed!")
    return True

def run_training(plan_name, custom_params=None):
    """运行训练"""
    if plan_name not in PLANS:
        print_error(f"Unknown plan: {plan_name}")
        print_info("Available plans: " + ", ".join(PLANS.keys()))
        return False
    
    # 获取基础参数
    config = PLANS[plan_name].copy()
    config.pop('description')
    config.pop('recommended')
    
    # 应用自定义参数
    if custom_params:
        config.update(custom_params)
    
    # 显示参数摘要
    print_header(f"Training Configuration - {plan_name}")
    print("Parameters:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print()
    
    # 构建命令
    cmd = [
        'python', 'train_det.py',
        '--epochs', str(config['epochs']),
        '--batch-size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--weight-decay', str(config['weight_decay']),
        '--growth-rate', str(config['growth_rate']),
        '--exp-name', f'optimized_{plan_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        '--log-freq', '2',
        '--plot-freq', '50',
    ]
    
    print_info("Starting training...")
    print_info(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print_warning("Training interrupted by user!")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='T型流体场预测模型优化训练启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Show available plans
  %(prog)s quick              # Run quick optimization plan
  %(prog)s advanced           # Run advanced optimization plan
  %(prog)s quick --epochs 250 # Run quick plan with custom epochs
  %(prog)s baseline           # Run baseline for comparison
        """
    )
    
    parser.add_argument('plan', nargs='?', default='show',
                       help='Training plan: baseline/quick/advanced/experimental/show')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--weight-decay', type=float, help='Override weight decay')
    parser.add_argument('--growth-rate', type=int, help='Override growth rate')
    
    args = parser.parse_args()
    
    # 检查环境
    if not validate_environment():
        return 1
    
    print()
    
    # 显示方案或运行训练
    if args.plan == 'show' or args.plan not in PLANS:
        show_plans()
        if args.plan != 'show':
            print_error(f"Unknown plan: {args.plan}")
            print_info(f"Available plans: {', '.join(PLANS.keys())}")
        return 0 if args.plan == 'show' else 1
    
    # 收集自定义参数
    custom_params = {}
    if args.epochs:
        custom_params['epochs'] = args.epochs
    if args.batch_size:
        custom_params['batch_size'] = args.batch_size
    if args.lr:
        custom_params['lr'] = args.lr
    if args.weight_decay:
        custom_params['weight_decay'] = args.weight_decay
    if args.growth_rate:
        custom_params['growth_rate'] = args.growth_rate
    
    # 显示方案信息
    plan_config = PLANS[args.plan]
    print_header(f"Training Plan: {args.plan}")
    print(f"Description: {plan_config['description']}")
    print()
    
    # 确认开始训练
    if custom_params:
        print_warning("Custom parameters will override plan defaults:")
        for key, value in custom_params.items():
            print(f"  {key}: {value}")
        print()
    
    response = input("Start training? (y/n): ").strip().lower()
    if response != 'y':
        print_info("Training cancelled.")
        return 0
    
    # 运行训练
    success = run_training(args.plan, custom_params)
    
    # 显示完成信息
    print()
    if success:
        print_success("-" * 70)
        print_success("Training completed successfully!")
        print_success("-" * 70)
        print_info("Next steps:")
        print_info("1. Check PDF files in experiments/deterministic/kle4225/.../training/predictions/")
        print_info("2. Compare row 1 (target) vs row 2 (prediction)")
        print_info("3. Check row 3 (error) - should be small")
        print_info("4. Review metrics in r2.pdf and rmse files")
        
        if args.plan == 'quick':
            print_info("\nFor better results, try: python run_training.py advanced")
    else:
        print_error("Training failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
