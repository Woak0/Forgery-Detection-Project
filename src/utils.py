"""
Utility functions for visualization and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def visualize_predictions(model, data_loader, device='cuda', num_samples=4):
    """Visualize model predictions"""
    model.eval()
    
    # Get a batch
    data, targets = next(iter(data_loader))
    data = data.to(device)
    
    with torch.no_grad():
        outputs = model(data)
        predictions = torch.sigmoid(outputs)
    
    # Move to CPU
    data = data.cpu()
    targets = targets.cpu()
    predictions = predictions.cpu()
    
    # Create figure
    num_samples = min(num_samples, len(data))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize RGB
        rgb = data[i, :3].numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb = rgb * std + mean
        rgb = np.clip(rgb.transpose(1, 2, 0), 0, 1)
        
        # ELA channel
        ela = data[i, 3].numpy()
        
        # Masks
        true_mask = targets[i, 0].numpy()
        pred_mask = (predictions[i, 0] > 0.5).numpy()
        
        # Calculate IoU
        intersection = (pred_mask * true_mask).sum()
        union = pred_mask.sum() + true_mask.sum() - intersection
        iou = intersection / (union + 1e-7)
        
        # Plot
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(ela, cmap='hot')
        axes[i, 1].set_title('ELA Channel')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(pred_mask, cmap='gray')
        axes[i, 3].set_title(f'Prediction (IoU: {iou:.3f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU plot
    axes[1].plot(epochs, history['train_iou'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_iou'], 'r-', label='Validation', linewidth=2)
    axes[1].set_title('IoU Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_model_performance(test_metrics: Dict[str, float]):
    """Analyze and interpret model performance"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\nPerformance Summary:")
    print(f"  IoU Score: {test_metrics['iou']:.3f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.3f}")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall: {test_metrics['recall']:.3f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    
    print("\nInterpretation:")
    
    if test_metrics['precision'] > test_metrics['recall']:
        print("  - Model is conservative: fewer false alarms but may miss some forgeries")
    else:
        print("  - Model is sensitive: detects most forgeries but may have false alarms")
    
    if test_metrics['iou'] > 0.5:
        print("  - Good spatial overlap with ground truth")
    elif test_metrics['iou'] > 0.3:
        print("  - Moderate detection capability")
    else:
        print("  - Needs improvement in localization")
    
    print("\nRecommendations:")
    
    if test_metrics['iou'] < 0.4:
        print("  1. Consider deeper architecture or more training data")
        print("  2. Adjust loss function weights")
    
    if test_metrics['recall'] < 0.5:
        print("  3. Increase Tversky loss beta parameter")
        print("  4. Use more aggressive augmentations")
    
    if test_metrics['precision'] < 0.5:
        print("  5. Apply post-processing to filter small detections")
        print("  6. Increase Tversky loss alpha parameter")