from PIL import Image, ImageChops, ImageEnhance
import io
import numpy as np
import os

def generate_ela(image_path, quality=90, scale=10):
    """
    Generates an Error Level Analysis (ELA) image for a given input image.
    
    Error Level Analysis works by re-saving an image at a known quality level
    and computing the difference between the original and re-saved versions.
    Regions with different compression histories will show different error levels.

    Args:
        image_path (str or PIL.Image): The file path to the original image or PIL Image object.
        quality (int): The JPEG quality level to use for re-compression (1-100).
                      Default 90 is recommended for forgery detection.
        scale (int): The factor by which to scale the pixel differences to make
                    them more visible. Default 10.

    Returns:
        PIL.Image.Image: The ELA image in grayscale.
    """
    # 1. Open the original image
    try:
        if isinstance(image_path, str):
            original_image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            original_image = image_path.convert('RGB')
        else:
            raise ValueError("image_path must be a file path string or PIL.Image object")
    except FileNotFoundError:
        print(f"Error: Could not find the image at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    # 2. Re-save the image at a specific JPEG quality into an in-memory buffer
    buffer = io.BytesIO()
    original_image.save(buffer, format='JPEG', quality=quality)
    
    # Rewind the buffer to the beginning to read from it
    buffer.seek(0)
    
    # 3. Open the re-compressed image from the buffer
    resaved_image = Image.open(buffer)

    # 4. Calculate the difference between the original and the re-saved image
    ela_image = ImageChops.difference(original_image, resaved_image)

    # 5. Scale the differences to make them easier to see
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    if max_diff == 0:
        # No differences found - return a black image
        return Image.new('L', original_image.size, 0)
    
    # Calculate scale factor
    scale_factor = min(255.0 / max_diff * (scale / 10.0), 255.0)
    
    # Apply scaling using point operation
    ela_image = ela_image.point(lambda i: min(int(i * scale_factor), 255))
    
    # Convert to grayscale for easier analysis
    ela_image = ela_image.convert('L')
    
    return ela_image


def generate_ela_tensor(image_tensor, quality=90, scale=10):
    """
    Generates ELA from a PyTorch tensor (for integration with data pipeline).
    
    Args:
        image_tensor: PyTorch tensor of shape [C, H, W] with values in [0, 1]
        quality: JPEG quality for re-compression
        scale: Scaling factor for differences
    
    Returns:
        numpy array of shape [H, W] with ELA values in [0, 1]
    """
    # Convert tensor to PIL Image
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Remove batch dimension if present
    
    # Denormalize if normalized with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    image_np = image_tensor.cpu().numpy()
    image_np = std * image_np + mean
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    # Convert to PIL
    image_np = image_np.transpose(1, 2, 0)
    pil_image = Image.fromarray(image_np, mode='RGB')
    
    # Generate ELA
    ela_image = generate_ela(pil_image, quality=quality, scale=scale)
    
    if ela_image is None:
        return np.zeros((image_tensor.shape[1], image_tensor.shape[2]))
    
    # Convert back to numpy array normalized to [0, 1]
    ela_np = np.array(ela_image).astype(np.float32) / 255.0
    
    return ela_np


def visualize_ela(image_path, quality=90, scale=10, save_path=None):
    """
    Visualize original image and its ELA side by side.
    
    Args:
        image_path: Path to image file
        quality: JPEG quality for ELA
        scale: Scaling factor
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    original = Image.open(image_path).convert('RGB')
    ela = generate_ela(image_path, quality=quality, scale=scale)
    
    if ela is None:
        print("Failed to generate ELA")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(ela, cmap='hot')
    axes[1].set_title(f"ELA (quality={quality}, scale={scale})")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()