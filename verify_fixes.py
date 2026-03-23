#!/usr/bin/env python3

"""
快速验证脚本 - 确认所有数据维度和类型修复都已到位
"""

import sys
import torch
from utils.load_data import load_data

print("=" * 70)
print(" 🔍 数据兼容性快速验证".center(70))
print("=" * 70)

try:
    print("\n[1/3] 加载数据...")
    train_loader, stats = load_data('./dataset/p_v_256.hdf5', 32)
    print("     ✓ 数据加载成功")
    
    print("\n[2/3] 检查批次数据...")
    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"     ✓ Input shape: {input_data.shape}")
            print(f"       dtype: {input_data.dtype}")
            print(f"     ✓ Target shape: {target_data.shape}")
            print(f"       dtype: {target_data.dtype}")
            
            # 验证期望的形状
            assert input_data.shape[0] == 32, "batch size mismatch"
            assert input_data.shape[1] == 1, "input channels should be 1"
            assert input_data.shape[2] == 137, "input height should be 137"
            assert input_data.shape[3] == 137, "input width should be 137"
            
            assert target_data.shape[0] == 32, "batch size mismatch"
            assert target_data.shape[1] == 4, "output channels should be 4"
            assert target_data.shape[2] == 137, "output height should be 137"
            assert target_data.shape[3] == 137, "output width should be 137"
            
            # 验证数据类型
            assert input_data.dtype == torch.float32, "input should be float32"
            assert target_data.dtype == torch.float32, "target should be float32"
            
            print("     ✓ 所有形状和类型验证通过")
            break
    
    print("\n[3/3] 测试模型兼容性...")
    from models.dense_ed import DenseED
    
    model = DenseED(
        in_channels=1,
        out_channels=4,
        blocks=[3, 6, 3],
        growth_rate=16,
        init_features=48,
        drop_rate=0.0,
        bn_size=8,
        bottleneck=False
    )
    
    # 测试前向传播
    dummy_input = torch.randn(2, 1, 137, 137, dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"     ✓ 模型前向传播成功")
    print(f"       Output shape: {output.shape}")
    assert output.shape == (2, 4, 137, 137), "output shape mismatch"
    
    print("\n" + "=" * 70)
    print(" ✅ 所有验证通过！可以开始训练了".center(70))
    print("=" * 70)
    print("\n推荐命令:")
    print("  python run_training.py quick")
    print("或")
    print("  python run_training.py advanced --epochs 500")
    
except Exception as e:
    print(f"\n❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
