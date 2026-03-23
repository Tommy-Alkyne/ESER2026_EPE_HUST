#!/bin/bash

# ============================================================================
# T型流体场预测模型优化训练脚本
# 使用改进的多阶段训练策略和增强的物理约束
#
# 使用方法:
#   chmod +x run_optimized_training.sh
#   ./run_optimized_training.sh [方案]
#
# 方案:
#   baseline   - 原始参数（对比用）
#   quick      - 快速优化方案（推荐新手）
#   advanced   - 高级优化方案（最佳效果）
#   custom     - 自定义参数
# ============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "working directory: $PWD"

# 检查必要文件
if [ ! -f "train_det.py" ]; then
    print_error "train_det.py not found！"
    exit 1
fi

if [ ! -f "dataset/p_v_256.hdf5" ]; then
    print_error "dataset/p_v_256.hdf5 not found!"
    exit 1
fi

# 确定运行方案
PLAN="${1:-quick}"

print_info "Selected plan: $PLAN"
print_info "================================"

# 方案参数定义
case $PLAN in
    baseline)
        print_warning "Baseline Plan - Original parameters (for comparison)"
        EPOCHS=200
        BATCH_SIZE=16
        LR=3e-3
        WEIGHT_DECAY=5e-4
        GROWTH_RATE=16
        DESCRIPTION="原始参数对比"
        ;;
        
    quick)
        print_success "Quick Optimization Plan - Recommended for first attempt"
        EPOCHS=300
        BATCH_SIZE=32
        LR=2e-3
        WEIGHT_DECAY=8e-4
        GROWTH_RATE=16
        DESCRIPTION="快速优化（新手推荐）"
        ;;
        
    advanced)
        print_success "Advanced Optimization Plan - Maximum improvement"
        EPOCHS=400
        BATCH_SIZE=64
        LR=1.5e-3
        WEIGHT_DECAY=1e-3
        GROWTH_RATE=24
        DESCRIPTION="高级优化（最佳效果）"
        ;;
        
    custom)
        print_info "Custom Parameters - Enter your settings"
        read -p "Number of epochs [300]: " EPOCHS
        EPOCHS=${EPOCHS:-300}
        read -p "Batch size [32]: " BATCH_SIZE
        BATCH_SIZE=${BATCH_SIZE:-32}
        read -p "Learning rate [2e-3]: " LR
        LR=${LR:-2e-3}
        read -p "Weight decay [8e-4]: " WEIGHT_DECAY
        WEIGHT_DECAY=${WEIGHT_DECAY:-8e-4}
        read -p "Growth rate [16]: " GROWTH_RATE
        GROWTH_RATE=${GROWTH_RATE:-16}
        DESCRIPTION="自定义参数"
        ;;
        
    *)
        print_error "Unknown plan: $PLAN"
        echo "Available plans: baseline, quick, advanced, custom"
        exit 1
        ;;
esac

print_info "Training Parameters:"
echo "  Epochs:       $EPOCHS"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Growth Rate:  $GROWTH_RATE"
echo "  Description:  $DESCRIPTION"

print_info "================================"
print_info "Starting training..."
print_info "Model will save best weights to: experiments/deterministic/kle4225/.../checkpoints/"
print_info "Predictions will be saved to: experiments/deterministic/kle4225/.../training/predictions/"
print_info ""

# 运行训练
python train_det.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --growth-rate "$GROWTH_RATE" \
    --exp-name "optimized_$PLAN" \
    --log-freq 2 \
    --plot-freq 50

print_success "================================"
print_success "Training completed!"
print_success "Plan used: $DESCRIPTION"
print_success "================================"

print_info "Next steps:"
echo "  1. Check PDF files in experiments/deterministic/kle4225/.../training/predictions/"
echo "  2. Compare row 1 (target) vs row 2 (prediction) - should be similar"
echo "  3. Check row 3 (error) - should be small"
echo "  4. View r2.pdf and other metrics"

# 建议后续操作
if [ "$PLAN" == "baseline" ]; then
    print_warning "Try 'quick' or 'advanced' plan for better results!"
elif [ "$PLAN" == "quick" ]; then
    print_info "Great! If results still need improvement, try 'advanced' plan."
    print_info "Run: ./run_optimized_training.sh advanced"
fi
