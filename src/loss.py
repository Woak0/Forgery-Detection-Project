import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    
    The Dice coefficient measures the overlap between predicted and ground truth masks.
    Dice Loss = 1 - Dice Coefficient
    
    This loss is particularly useful for imbalanced datasets where the positive class
    (forged regions) is much smaller than the negative class (authentic regions).
    
    Args:
        smooth (float): Smoothing constant to avoid division by zero. Default: 1.0
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits from model (before sigmoid) [B, 1, H, W]
            targets: Ground truth binary masks [B, 1, H, W]
        
        Returns:
            dice_loss: Scalar loss value
        """
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors to [B*H*W]
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        
        # Dice coefficient: 2 * |X ∩ Y| / (|X| + |Y|)
        dice_coefficient = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        # Return Dice Loss
        return 1.0 - dice_coefficient


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    
    Focal Loss = -α * (1 - p)^γ * log(p)
    
    Args:
        alpha (float): Weighting factor for positive class. Default: 0.25
        gamma (float): Focusing parameter. Higher values down-weight easy examples more. Default: 2.0
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits [B, 1, H, W]
            targets: Ground truth [B, 1, H, W]
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if y=1, else 1-p
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with configurable weights.
    
    Default: BCE + Dice Loss
    Optional: Add Focal Loss for hard example mining
    
    Args:
        bce_weight (float): Weight for BCE loss. Default: 0.5
        dice_weight (float): Weight for Dice loss. Default: 0.5
        focal_weight (float): Weight for Focal loss. Default: 0.0 (disabled)
        use_focal (bool): Whether to use Focal loss. Default: False
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, focal_weight=0.0, use_focal=False):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.use_focal = use_focal
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
        if self.use_focal:
            self.focal = FocalLoss()
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits [B, 1, H, W]
            targets: Ground truth [B, 1, H, W]
        
        Returns:
            combined_loss: Weighted sum of losses
            loss_dict: Dictionary with individual loss values for logging
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        combined = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'total': combined.item()
        }
        
        if self.use_focal:
            focal_loss = self.focal(inputs, targets)
            combined = combined + self.focal_weight * focal_loss
            loss_dict['focal'] = focal_loss.item()
            loss_dict['total'] = combined.item()
        
        return combined, loss_dict


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss, also known as Jaccard Loss.
    
    IoU = |X ∩ Y| / |X ∪ Y|
    IoU Loss = 1 - IoU
    
    Args:
        smooth (float): Smoothing constant. Default: 1.0
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - iou


# Utility function to get loss by name
def get_loss_function(loss_name='combined', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: One of ['bce', 'dice', 'focal', 'iou', 'combined']
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function instance
    """
    loss_dict = {
        'bce': nn.BCEWithLogitsLoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'iou': IoULoss,
        'combined': CombinedLoss
    }
    
    if loss_name not in loss_dict:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(loss_dict.keys())}")
    
    return loss_dict[loss_name](**kwargs)