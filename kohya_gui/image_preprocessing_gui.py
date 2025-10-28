import gradio as gr
import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
import cv2
import re
import math

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import (
    get_folder_path,
    folder_symbol,
)
from .custom_logging import setup_logging

log = setup_logging()

# Add sd-scripts to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "sd-scripts"))

# Import Kohya's bucket logic
from library.model_util import make_bucket_resolutions


def gradio_image_preprocessing_gui_tab(headless: bool = False, config: KohyaSSGUIConfig = {}):
    """
    Create the Image Preprocessing tab for visualizing Kohya bucket processing
    
    Args:
        headless: Whether to run in headless mode
        config: Configuration object
    """
    
    def natural_sort_key(path, regex=re.compile(r'(\d+)')):
        """Natural sort key function for cross-platform filename sorting"""
        return [int(text) if text.isdigit() else text.lower() for text in regex.split(str(path))]
    
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        """Calculate crop coordinates according to Kohya's preprocessing"""
        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucket is wider → match height
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            # bucket is taller → match width
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom
    
    def resize_and_crop_image(image, bucket_reso):
        """Resize and crop image according to Kohya's logic"""
        original_size = image.size  # (width, height)
        
        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = original_size[0] / original_size[1]
        
        # Calculate scale factor
        if bucket_ar > image_ar:
            # bucket is wider → match height
            scale = bucket_reso[1] / original_size[1]
        else:
            # bucket is taller → match width
            scale = bucket_reso[0] / original_size[0]
        
        # Resize image
        resized_size = (int(original_size[0] * scale + 0.5), int(original_size[1] * scale + 0.5))
        
        if scale > 1:
            resized_image = image.resize(resized_size, Image.LANCZOS)
        else:
            resized_image = image.resize(resized_size, Image.LANCZOS)
        
        # Crop to bucket resolution
        crop_left, crop_top, crop_right, crop_bottom = get_crop_ltrb(bucket_reso, original_size)
        crop_box = (int(crop_left), int(crop_top), int(crop_right), int(crop_bottom))
        cropped_image = resized_image.crop(crop_box)
        
        return cropped_image
    
    def process_images(
        input_folder: str,
        output_folder: str,
        architecture: str,
        enable_bucket: bool,
        max_resolution_width: int,
        max_resolution_height: int,
        fix_exif_orientation: bool,
        progress=gr.Progress()
    ) -> Tuple[str, list]:
        """
        Process images using Kohya's bucket logic
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save processed images
            architecture: Model architecture (sdxl, flux, sd3, sd1, etc.)
            enable_bucket: Whether to use bucket resolution selection
            max_resolution_width: Max resolution width
            max_resolution_height: Max resolution height
            fix_exif_orientation: Whether to correct EXIF orientation metadata
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status message, list of processed files)
        """
        try:
            # Validate inputs
            if not input_folder or not os.path.exists(input_folder):
                return "Error: Input folder does not exist", []
            
            if not output_folder:
                return "Error: Output folder not specified", []
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Get list of image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(input_folder).glob(f'*{ext}'))
                image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
            
            # Deduplicate (Windows filesystem is case-insensitive, so glob patterns may match same files)
            # Use natural sort for cross-platform consistent ordering (e.g., file2.jpg before file10.jpg)
            image_files = sorted(set(image_files), key=natural_sort_key)
            
            if not image_files:
                return "Error: No image files found in input folder", []
            
            processed_files = []
            bucket_stats = {}
            
            # Map architecture to resolution steps
            ARCHITECTURE_RESO_STEPS = {
                "sdxl": 32,
                "flux": 32,
                "sd3": 8,
                "sd1": 64,
                "sd2": 64,
            }
            reso_steps = ARCHITECTURE_RESO_STEPS.get(architecture.lower(), 64)
            
            # Generate bucket resolutions if enabled
            buckets = []
            if enable_bucket:
                max_reso = (max_resolution_width, max_resolution_height)
                buckets = make_bucket_resolutions(max_reso, min_size=256, max_size=2048, divisible=reso_steps)
                buckets_sorted = sorted(buckets)
            
            total_images = len(image_files)
            
            for idx, image_path in enumerate(image_files):
                progress(idx / total_images, desc=f"Processing {image_path.name}")
                
                try:
                    # Load image
                    image = Image.open(image_path)
                    
                    # Apply EXIF orientation correction if enabled
                    if fix_exif_orientation:
                        image = ImageOps.exif_transpose(image)
                    
                    original_size = image.size  # (width, height)
                    
                    # Determine bucket resolution
                    if enable_bucket and buckets:
                        # Find nearest bucket by aspect ratio
                        aspect_ratio = original_size[0] / original_size[1]
                        aspect_ratios = np.array([w / h for w, h in buckets_sorted])
                        ar_errors = np.abs(aspect_ratios - aspect_ratio)
                        bucket_id = ar_errors.argmin()
                        bucket_reso = buckets_sorted[bucket_id]
                    else:
                        bucket_reso = (max_resolution_width, max_resolution_height)
                    
                    # Track bucket usage
                    bucket_key = f"{bucket_reso[0]}x{bucket_reso[1]}"
                    bucket_stats[bucket_key] = bucket_stats.get(bucket_key, 0) + 1
                    
                    # Process image using Kohya's resize and crop logic
                    processed_image = resize_and_crop_image(image, bucket_reso)
                    
                    # Save processed image
                    output_path = os.path.join(output_folder, image_path.name)
                    processed_image.save(output_path, quality=95)
                    
                    processed_files.append({
                        'original': f"{original_size[0]}x{original_size[1]}",
                        'bucket': bucket_key,
                        'file': image_path.name
                    })
                    
                except Exception as e:
                    log.error(f"Error processing {image_path.name}: {e}")
                    continue
            
            # Generate status message
            status_parts = [
                f"✓ Processed {len(processed_files)} images",
                f"\n\nBucket Distribution:"
            ]
            
            for bucket, count in sorted(bucket_stats.items()):
                percentage = (count / len(processed_files)) * 100
                status_parts.append(f"  {bucket}: {count} images ({percentage:.1f}%)")
            
            status_message = "\n".join(status_parts)
            
            return status_message, processed_files
            
        except Exception as e:
            log.error(f"Error in process_images: {e}")
            return f"Error: {str(e)}", []
    
    with gr.Column():
        gr.Markdown(
            """
            # Image Preprocessing Tool
            
            This tool demonstrates how Kohya processes images with bucket resolution selection.
            It takes your images and processes them exactly as Kohya would during training.
            
            **How it works:**
            - If bucketing is enabled, images are assigned to the nearest bucket resolution based on aspect ratio
            - Images are resized (maintaining aspect ratio) then center-cropped to the bucket resolution
            - Processed images are saved to the output folder with the same filenames
            
            **Note:** This tool can optionally correct EXIF orientation metadata. Kohya does NOT do this automatically during training.
            """
        )
        
        with gr.Row():
            input_folder = gr.Textbox(
                label="Input Images Folder",
                placeholder="Path to folder containing images",
                interactive=True,
            )
            input_folder_button = gr.Button(
                folder_symbol,
                elem_id="open_folder",
                elem_classes=["tool"],
            )
            
            output_folder = gr.Textbox(
                label="Output Folder",
                placeholder="Path to save processed images",
                interactive=True,
            )
            output_folder_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_save",
                elem_classes=["tool"],
            )
        
        with gr.Row():
            architecture = gr.Dropdown(
                label="Architecture",
                choices=[
                    ("sdxl - SDXL", "sdxl"),
                    ("flux - Flux", "flux"),
                    ("sd3 - SD3", "sd3"),
                    ("sd1 - SD 1.x", "sd1"),
                    ("sd2 - SD 2.x", "sd2"),
                ],
                value="flux",
                info="Model architecture (determines resolution steps)"
            )
            
            enable_bucket = gr.Checkbox(
                label="Enable Bucketing",
                value=True,
                info="When enabled, images are assigned to nearest bucket resolution based on aspect ratio"
            )
            
            fix_exif_orientation = gr.Checkbox(
                label="Fix EXIF Orientation",
                value=False,
                info="Correct image orientation based on EXIF metadata (Kohya does NOT do this automatically)"
            )
            
            resolution_width = gr.Number(
                label="Max Resolution Width",
                value=1024,
                minimum=256,
                maximum=2048,
                step=64,
                info="Maximum resolution width for bucket generation"
            )
            
            resolution_height = gr.Number(
                label="Max Resolution Height",
                value=1024,
                minimum=256,
                maximum=2048,
                step=64,
                info="Maximum resolution height for bucket generation"
            )
        
        with gr.Row():
            process_button = gr.Button(
                value="Process Images",
                variant="primary",
            )
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                lines=10,
            )
        
        with gr.Row():
            file_list = gr.JSON(
                label="Processed Files",
                value=[],
            )
        
        # Button events
        input_folder_button.click(
            fn=lambda x: get_folder_path(x),
            inputs=[input_folder],
            outputs=[input_folder],
            show_progress=False,
        )
        
        output_folder_button.click(
            fn=lambda x: get_folder_path(x),
            inputs=[output_folder],
            outputs=[output_folder],
            show_progress=False,
        )
        
        process_button.click(
            fn=process_images,
            inputs=[
                input_folder,
                output_folder,
                architecture,
                enable_bucket,
                resolution_width,
                resolution_height,
                fix_exif_orientation,
            ],
            outputs=[status_output, file_list],
        )

