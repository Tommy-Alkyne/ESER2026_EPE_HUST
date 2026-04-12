"""
物理约束损失函数模块，针对T型流体场的特殊设计 (加入 PINN Navier-Stokes 约束)
"""

import torch
import torch.nn.functional as F


def get_diff_kernels(device):
    """生成用于计算一阶有限差分的固定卷积核"""
    kernel_x = torch.tensor([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    kernel_y = torch.tensor([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    return kernel_x, kernel_y


def get_laplacian_kernel(device):
    """生成用于计算二阶导数(Laplacian)的固定卷积核"""
    kernel_laplace = torch.tensor([[0, 1, 0], 
                                   [1, -4, 1], 
                                   [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
    return kernel_laplace


def compute_divergence(velocity_field, dx=1.0, dy=1.0):
    """使用F.conv2d向量化计算散度，padding=1保留边界信息"""
    if velocity_field.shape[1] < 2:
        return None
        
    is_5d = velocity_field.dim() == 5
    if is_5d:
        B, C, D, H, W = velocity_field.shape
        velocity_field = velocity_field.reshape(B * D, C, H, W)
        
    vx = velocity_field[:, 0:1, :, :]
    vy = velocity_field[:, 1:2, :, :]
    
    kernel_x, kernel_y = get_diff_kernels(velocity_field.device)
    
    # padding=1 避免了原版尺寸缩小的问题
    dvx_dx = F.conv2d(vx, kernel_x, padding=1) / dx
    dvy_dy = F.conv2d(vy, kernel_y, padding=1) / dy
    
    divergence = dvx_dx + dvy_dy
    if is_5d:
        divergence = divergence.reshape(B, 1, D, H, W)
    return divergence


def divergence_loss(output, target_divergence_penalty=1.0, dx=1.0, dy=1.0):
    """
    连续性方程损失：鼓励预测的速度场满足 div(v) ≈ 0
    """
    if output.shape[1] < 2:
        return torch.tensor(0.0, device=output.device)
    
    divergence = compute_divergence(output, dx, dy)
    if divergence is None:
        return torch.tensor(0.0, device=output.device)
    
    # 目标是让散度尽可能接近0
    loss = torch.mean(divergence ** 2) * target_divergence_penalty
    return loss


def navier_stokes_loss(output, mask=None, rho=1.0, nu=0.01, dx=1.0, dy=1.0):
    """
    [新增] 基于有限差分的 Navier-Stokes 动量方程物理约束损失 (Discrete PINN)
    假设输出通道顺序为: [Vx, Vy, P] 
    """
    # 如果输出没有包含压力P，则无法完整计算动量方程，直接返回0
    if output.shape[1] < 3:
        return torch.tensor(0.0, device=output.device)

    is_5d = output.dim() == 5
    if is_5d:
        B, C, D, H, W = output.shape
        output = output.reshape(B * D, C, H, W)
        if mask is not None:
            mask = mask.expand(B, C, D, H, W) if mask.dim() == 5 else mask.expand(B, 1, D, H, W)
            mask = mask.reshape(B * D, -1, H, W)[:, 0:1, :, :] # keep 1 channel

    vx = output[:, 0:1, :, :]
    vy = output[:, 1:2, :, :]
    p  = output[:, 2:3, :, :]

    kernel_x, kernel_y = get_diff_kernels(output.device)
    kernel_laplace = get_laplacian_kernel(output.device)

    # 计算一阶导数
    dvx_dx = F.conv2d(vx, kernel_x, padding=1) / dx
    dvx_dy = F.conv2d(vx, kernel_y, padding=1) / dy
    dvy_dx = F.conv2d(vy, kernel_x, padding=1) / dx
    dvy_dy = F.conv2d(vy, kernel_y, padding=1) / dy
    dp_dx  = F.conv2d(p, kernel_x, padding=1) / dx
    dp_dy  = F.conv2d(p, kernel_y, padding=1) / dy

    # 计算二阶导数 (Laplacian = d2/dx2 + d2/dy2)
    laplace_vx = F.conv2d(vx, kernel_laplace, padding=1) / (dx * dy)
    laplace_vy = F.conv2d(vy, kernel_laplace, padding=1) / (dx * dy)

    # 稳态不可压缩 Navier-Stokes 动量方程残差
    # X方向: u(du/dx) + v(du/dy) + (1/rho)(dp/dx) - nu(nabla^2 u) = 0
    momentum_x = vx * dvx_dx + vy * dvx_dy + (1.0 / rho) * dp_dx - nu * laplace_vx
    
    # Y方向: u(dv/dx) + v(dv/dy) + (1/rho)(dp/dy) - nu(nabla^2 v) = 0
    momentum_y = vx * dvy_dx + vy * dvy_dy + (1.0 / rho) * dp_dy - nu * laplace_vy

    if mask is not None:
        momentum_x = momentum_x * mask
        momentum_y = momentum_y * mask

    loss_ns = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2)
    
    if is_5d and mask is not None:
        mask = mask.reshape(B, 1, D, H, W)
        
    return loss_ns


def smoothness_loss(tensor, weight=1.0):
    """向量化的平滑度损失"""
    
    is_5d = tensor.dim() == 5
    if is_5d:
        B, C, D, H, W = tensor.shape
        tensor = tensor.reshape(B * D, C, H, W)
        
    kernel_x, kernel_y = get_diff_kernels(tensor.device)
    
    # 分组卷积以处理多通道(如P, Vx, Vy)
    channels = tensor.shape[1]
    kernel_x_group = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y_group = kernel_y.repeat(channels, 1, 1, 1)
    
    dx = F.conv2d(tensor, kernel_x_group, padding=1, groups=channels)
    dy = F.conv2d(tensor, kernel_y_group, padding=1, groups=channels)
    
    smoothness = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    return smoothness * weight


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
                         div_weight=0.1, smooth_weight=0.01, ns_weight=0.05, 
                         rho=1.0, nu=0.01, dx=1.0, dy=1.0):
    """
    组合的物理约束损失，包含 PINN Navier-Stokes 约束，随着训练进展动态调整权重
    """
    base_loss = reconstruction_loss(output, target, mask=mask, use_weighted=True)
    
    # 连续性方程约束（散度）
    div_loss = divergence_loss(output, target_divergence_penalty=1.0, dx=dx, dy=dy)
    
    # PINN Navier-Stokes 动量方程约束
    ns_loss = navier_stokes_loss(output, mask=mask, rho=rho, nu=nu, dx=dx, dy=dy)
    
    # 平滑性约束
    smooth_loss = smoothness_loss(output, weight=1.0)
    
    # 动态权重调整：在训练早期侧重数据重建，中后期逐渐增强物理约束的权重
    progress = epoch / total_epochs
    div_weight_dynamic = div_weight * (1.0 + progress)
    ns_weight_dynamic = ns_weight * (1.0 + progress)
    smooth_weight_dynamic = smooth_weight * (1.0 - 0.5 * progress) # 平滑约束在使用PINN精确方程后可以适当减弱
    
    total_loss = base_loss + (div_weight_dynamic * div_loss) + (ns_weight_dynamic * ns_loss) + (smooth_weight_dynamic * smooth_loss)
    
    return total_loss, {
        'base': base_loss.item(),
        'div': div_loss.item(),
        'ns': ns_loss.item(),
        'smooth': smooth_loss.item()
    }