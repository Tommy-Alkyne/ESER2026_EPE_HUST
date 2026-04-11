"""
物理约束损失函数模块，针对T型流体场的特殊设计
"""

import torch
import torch.nn.functional as F


def get_diff_kernels(device):
    """生成用于计算有限差分的固定卷积核"""
    kernel_x = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    kernel_y = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    return kernel_x, kernel_y



def compute_divergence(velocity_field):
    """使用F.conv2d向量化计算散度，padding=1保留边界信息"""
    if velocity_field.shape[1] < 2:
        return None
        
    vx = velocity_field[:, 0:1, :, :]
    vy = velocity_field[:, 1:2, :, :]
    
    kernel_x, kernel_y = get_diff_kernels(velocity_field.device)
    
    # padding=1 避免了原版尺寸缩小的问题
    dvx_dx = F.conv2d(vx, kernel_x, padding=1)
    dvy_dy = F.conv2d(vy, kernel_y, padding=1)
    
    return dvx_dx + dvy_dy


def divergence_loss(output, target_divergence_penalty=1.0):
    """
    连续性方程损失：鼓励预测的速度场满足 div(v) ≈ 0
    """
    if output.shape[1] < 2:
        return torch.tensor(0.0, device=output.device)
    
    divergence = compute_divergence(output)
    if divergence is None:
        return torch.tensor(0.0, device=output.device)
    
    # 目标是让散度尽可能接近0
    loss = torch.mean(divergence ** 2) * target_divergence_penalty
    return loss


def smoothness_loss(tensor, weight=1.0):
    """向量化的平滑度损失"""
    kernel_x, kernel_y = get_diff_kernels(tensor.device)
    
    # 分组卷积以处理多通道(如P, Vx, Vy)
    channels = tensor.shape[1]
    kernel_x_group = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y_group = kernel_y.repeat(channels, 1, 1, 1)
    
    dx = F.conv2d(tensor, kernel_x_group, padding=1, groups=channels)
    dy = F.conv2d(tensor, kernel_y_group, padding=1, groups=channels)
    
    smoothness = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    return smoothness * weight


def boundary_gradient_loss(output, target, boundary_mask, weight=1.0):
    """
    边界梯度约束：在T型边界处施加更强的约束
    """
    if boundary_mask is None:
        return torch.tensor(0.0, device=output.device)
    
    # 在边界处计算梯度
    boundary_output = output * boundary_mask
    boundary_target = target * boundary_mask
    
    grad_loss = F.l1_loss(boundary_output, boundary_target, reduction='mean')
    return grad_loss * weight


def reconstruction_loss(output, target, mask=None, use_weighted=False):
    """
    改进的重建损失，结合MSE和L1损失
    """
    if mask is not None:
        output = output * mask
        target = target * mask
    
    mse_loss = F.mse_loss(output, target, reduction='mean')
    l1_loss = F.l1_loss(output, target, reduction='mean')
    
    if use_weighted:
        # 加权组合：MSE捕捉大误差，L1对异常值更鲁棒
        loss = 0.6 * mse_loss + 0.4 * l1_loss
    else:
        loss = mse_loss + l1_loss
    
    return loss


def combined_physics_loss(output, target, mask=None, epoch=1, total_epochs=200, 
                         div_weight=0.1, smooth_weight=0.05):
    """
    组合的物理约束损失，随着训练进展动态调整权重
    """
    base_loss = reconstruction_loss(output, target, mask=mask, use_weighted=True)
    
    # 连续性方程约束（随着训练增强）
    div_loss = divergence_loss(output, target_divergence_penalty=1.0)
    
    # 平滑性约束
    smooth_loss = smoothness_loss(output, weight=smooth_weight)
    
    # 动态权重调整
    progress = epoch / total_epochs
    div_weight_dynamic = div_weight * (1.0 + progress)  # 训练后期增强
    
    total_loss = base_loss + div_weight_dynamic * div_loss + smooth_loss
    
    return total_loss, {
        'base': base_loss.item(),
        'div': div_loss.item(),
        'smooth': smooth_loss.item()
    }
