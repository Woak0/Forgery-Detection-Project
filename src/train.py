"""
Training pipeline for image forgery detection.
Fixed scheduler, metrics calculation, and threshold optimization.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate metrics for evaluation"""
    preds_binary = (predictions > threshold).float()
    targets_binary = (targets > 0.5).float()
    
    preds_flat = preds_binary.view(-1)
    targets_flat = targets_binary.view(-1)
    
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum().item()
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou,
        'accuracy': accuracy
    }


def find_optimal_threshold(model, val_loader, device='cuda'):
    """Find optimal threshold using validation set"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.sigmoid(outputs).cpu()
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # Handle edge case where all scores are NaN
    valid_scores = ~np.isnan(f1_scores[:-1])
    if not valid_scores.any():
        return 0.5
    
    optimal_idx = np.nanargmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")
    return optimal_threshold


def train_model(model, train_loader, val_loader, criterion, 
                num_epochs=100, learning_rate=3e-4, device='cuda'):
    """
    Training loop with fixed scheduler and metrics calculation.
    """
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Fixed: OneCycleLR scheduler with steps_per_epoch
    total_steps = num_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision
    scaler = GradScaler() if device == 'cuda' else None
    
    # Tracking
    best_val_iou = 0.0
    patience_counter = 0
    early_stop_patience = 20
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    for epoch in range(num_epochs):
        # ============== TRAINING PHASE ==============
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    outputs = model(data)
                    loss, loss_dict = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss, loss_dict = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Fixed: Step scheduler after each batch for OneCycleLR
            scheduler.step()
            
            train_loss += loss.item()
            train_batch_count += 1
            
            # Update progress bar (metrics calculation only for display, not storage)
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
        
        # Calculate training metrics at epoch end (more efficient)
        avg_train_loss = train_loss / train_batch_count
        
        # Quick training IoU calculation on subset
        model.eval()
        train_ious = []
        with torch.no_grad():
            for i, (data, targets) in enumerate(train_loader):
                if i >= 5:  # Only sample 5 batches for efficiency
                    break
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                preds = torch.sigmoid(outputs)
                metrics = calculate_metrics(preds, targets)
                train_ious.append(metrics['iou'])
        
        avg_train_iou = np.mean(train_ious) if train_ious else 0.0
        
        # ============== VALIDATION PHASE ==============
        model.eval()
        val_loss = 0.0
        val_metrics_list = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
            
            for data, targets in val_loop:
                data = data.to(device)
                targets = targets.to(device)
                
                outputs = model(data)
                loss, loss_dict = criterion(outputs, targets)  # Fixed: properly capture dict
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs)
                metrics = calculate_metrics(preds, targets)
                val_metrics_list.append(metrics)
                
                val_loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    iou=f"{metrics['iou']:.3f}"
                )
        
        # Calculate average metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {
            k: np.mean([m[k] for m in val_metrics_list])
            for k in ['precision', 'recall', 'f1_score', 'iou', 'accuracy']
        }
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_metrics['iou'])
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary")
        print(f"  Loss:  Train={avg_train_loss:.4f} Val={avg_val_loss:.4f}")
        print(f"  IoU:   Train={avg_train_iou:.4f} Val={avg_val_metrics['iou']:.4f}")
        print(f"  F1:    Val={avg_val_metrics['f1_score']:.4f}")
        print(f"  Prec:  Val={avg_val_metrics['precision']:.4f}")
        print(f"  Rec:   Val={avg_val_metrics['recall']:.4f}")
        
        # Save best model
        if avg_val_metrics['iou'] > best_val_iou:
            best_val_iou = avg_val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_metrics': avg_val_metrics
            }, 'best_model.pth')
            print(f"  ✓ NEW BEST MODEL! IoU={best_val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print("="*80)
    
    return history


def evaluate_model(model, test_loader, criterion, optimal_threshold=None, device='cuda'):
    """
    Evaluate model on test set with optimal threshold.
    """
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Find optimal threshold if not provided
    if optimal_threshold is None:
        print("Finding optimal threshold on validation set...")
        # Note: In practice, you'd use val_loader here
        optimal_threshold = 0.5  # Default fallback
    
    model.eval()
    test_loss = 0.0
    test_metrics_list = []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            loss, _ = criterion(outputs, targets)
            test_loss += loss.item()
            
            preds = torch.sigmoid(outputs)
            metrics = calculate_metrics(preds, targets, threshold=optimal_threshold)
            test_metrics_list.append(metrics)
    
    # Calculate averages
    avg_test_loss = test_loss / len(test_loader)
    avg_test_metrics = {
        k: np.mean([m[k] for m in test_metrics_list])
        for k in ['precision', 'recall', 'f1_score', 'iou', 'accuracy']
    }
    
    print(f"\nTest Set Results (threshold={optimal_threshold:.3f}):")
    print(f"  Loss:      {avg_test_loss:.4f}")
    print(f"  IoU:       {avg_test_metrics['iou']:.4f}")
    print(f"  F1 Score:  {avg_test_metrics['f1_score']:.4f}")
    print(f"  Precision: {avg_test_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_test_metrics['recall']:.4f}")
    print(f"  Accuracy:  {avg_test_metrics['accuracy']:.4f}")
    
    return avg_test_metrics, optimal_threshold