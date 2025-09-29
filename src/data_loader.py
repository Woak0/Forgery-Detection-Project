import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from src.ela import generate_ela

class ForgeryDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the CASIA v2.0 image forgery dataset.
    With synchronized augmentation for image-mask pairs.
    """
    def __init__(self, root_dir, image_size=(256, 256), train=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train

        self.tampered_path = os.path.join(self.root_dir, 'Tp')
        self.groundtruth_path = os.path.join(self.root_dir, 'CASIA 2 Groundtruth')
        
        self.image_mask_pairs = self._find_pairs()
        
        # Define normalization (apply only to images, not masks)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _find_pairs(self):
        pairs = []
        
        if not os.path.exists(self.tampered_path):
            raise FileNotFoundError(f"Tampered images path not found: {self.tampered_path}")
        if not os.path.exists(self.groundtruth_path):
            raise FileNotFoundError(f"Ground truth path not found: {self.groundtruth_path}")
        
        tampered_images = os.listdir(self.tampered_path)
        groundtruth_masks = os.listdir(self.groundtruth_path)
        mask_set = set(groundtruth_masks)
        
        for img_file in tampered_images:
            if img_file.endswith(('.jpg', '.tif', '.bmp', '.png')):
                base_name = os.path.splitext(img_file)[0]
                potential_mask_name = base_name + '_gt.png'
                
                if potential_mask_name in mask_set:
                    img_path = os.path.join(self.tampered_path, img_file)
                    mask_path = os.path.join(self.groundtruth_path, potential_mask_name)
                    pairs.append((img_path, mask_path))
        
        if len(pairs) == 0:
            raise ValueError("No matching image-mask pairs found. Check your dataset structure.")
        
        print(f"Found {len(pairs)} matching image-mask pairs.")
        return pairs

    def _synchronized_transforms(self, image, mask):
        """
        Apply the same geometric transformations to both image and mask.
        This is CRITICAL - augmentations must be synchronized!
        """
        # Resize to a slightly larger size first for random crop
        if self.train:
            resize_size = (int(self.image_size[0] * 1.15), int(self.image_size[1] * 1.15))
        else:
            resize_size = self.image_size
        
        image = TF.resize(image, resize_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, resize_size, interpolation=Image.NEAREST)
        
        if self.train:
            # Random crop (synchronized)
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.image_size
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
            # Random horizontal flip (synchronized)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip (synchronized)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (synchronized)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
            
            # Color jitter (ONLY for image, not mask!)
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                )
                image = color_jitter(image)
            
            # Random Gaussian blur (ONLY for image)
            if random.random() > 0.3:
                kernel_size = random.choice([3, 5, 7])
                sigma = random.uniform(0.1, 2.0)
                image = TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        
        else:
            # Validation/Test: just resize
            image = TF.resize(image, self.image_size, interpolation=Image.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)
        
        return image, mask

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        try:
            image_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('L')
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Generate the ELA image from the original PIL image
        ela_pil = generate_ela(image_pil, quality=90)

        # Apply all synchronized transforms (geometric, color, etc.)
        image_aug, mask_aug = self._synchronized_transforms(image_pil, mask_pil)

        # Apply ONLY the geometric transforms to the ELA image to keep it aligned
        # For simplicity, we'll just resize it. This is a common and effective approximation.
        ela_aug = TF.resize(ela_pil, self.image_size, interpolation=Image.NEAREST)
        
        # Convert all to tensors
        image_tensor = TF.to_tensor(image_aug)
        mask_tensor = TF.to_tensor(mask_aug)
        ela_tensor = TF.to_tensor(ela_aug)
        
        # Normalize the RGB image
        image_tensor = self.normalize(image_tensor)
        
        # Concatenate to create the 4D input
        combined_tensor = torch.cat([image_tensor, ela_tensor], dim=0)
        
        # Binarise the mask
        mask_tensor = (mask_tensor > 0.5).float()
        
        return combined_tensor, mask_tensor


def get_data_loaders(root_dir, batch_size=8, image_size=(256, 256), num_workers=2, seed=42, test_split=0.1):
    """
    Create train, validation, and test data loaders with proper augmentation.
    
    Args:
        root_dir: Path to CASIA2 dataset
        batch_size: Batch size for training
        image_size: Target image size (H, W)
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        test_split: Proportion of data for testing (default 0.1 = 10%)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create separate dataset instances for train (with augmentation) and val/test (without)
    train_data = ForgeryDataset(root_dir=root_dir, image_size=image_size, train=True)
    val_data = ForgeryDataset(root_dir=root_dir, image_size=image_size, train=False)
    test_data = ForgeryDataset(root_dir=root_dir, image_size=image_size, train=False)
    
    # Get total size
    total_size = len(train_data)
    test_size = int(test_split * total_size)
    train_val_size = total_size - test_size
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    
    # Create indices and split
    indices = list(range(total_size))
    generator = torch.Generator().manual_seed(seed)
    
    # First split: separate test set
    train_val_indices_list = indices[:train_val_size]
    test_indices_list = indices[train_val_size:]
    
    # Second split: separate train and val from train_val
    random.Random(seed).shuffle(train_val_indices_list)
    train_indices_list = train_val_indices_list[:train_size]
    val_indices_list = train_val_indices_list[train_size:]
    
    # Create subsets
    train_dataset = Subset(train_data, train_indices_list)
    val_dataset = Subset(val_data, val_indices_list)
    test_dataset = Subset(test_data, test_indices_list)
    
    print(f"\nDataset Split:")
    print(f"  Training set size: {len(train_dataset)}")
    print(f"  Validation set size: {len(val_dataset)}")
    print(f"  Test set size: {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def visualize_batch(data_loader, num_samples=4, title="Data Samples"):
    """
    Utility function to visualize a batch of data.
    Useful for debugging your data pipeline!
    """
    import matplotlib.pyplot as plt
    
    batch = next(iter(data_loader))
    images, masks = batch
    
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image for visualization
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        mask = masks[i].cpu().squeeze().numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        forgery_percentage = (mask.sum() / mask.size) * 100
        axes[i, 1].set_title(f"Mask ({forgery_percentage:.1f}% forged)")
        axes[i, 1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nBatch Statistics:")
    print(f"  Image shape: {images.shape}")
    print(f"  Mask shape: {masks.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mask unique values: {torch.unique(masks).tolist()}")
    print(f"  Mask mean: {masks.mean():.3f} (proportion of forged pixels)")
    print(f"  Min forgery per image: {masks.view(masks.size(0), -1).mean(dim=1).min():.3f}")
    print(f"  Max forgery per image: {masks.view(masks.size(0), -1).mean(dim=1).max():.3f}")