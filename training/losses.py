import torch
import torch.nn.functional as F


def compute_gradient_loss(output, target, mask=None):
    if output.dim() != 5:
        raise ValueError('Unsupported output dim {}'.format(output.dim()))

    grad_output_z = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
    grad_output_y = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
    grad_output_x = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]

    grad_target_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    grad_target_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    grad_target_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

    if mask is not None:
        mask_z = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
        mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
        mask_x = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]

        grad_output_z = grad_output_z * mask_z
        grad_output_y = grad_output_y * mask_y
        grad_output_x = grad_output_x * mask_x

        grad_target_z = grad_target_z * mask_z
        grad_target_y = grad_target_y * mask_y
        grad_target_x = grad_target_x * mask_x

    loss_z = F.l1_loss(grad_output_z, grad_target_z, reduction='sum')
    loss_y = F.l1_loss(grad_output_y, grad_target_y, reduction='sum')
    loss_x = F.l1_loss(grad_output_x, grad_target_x, reduction='sum')
    return loss_z + loss_y + loss_x
