#!/usr/bin/env python3
"""
Create video from TIF image sequence with global normalization.

This script processes a directory of 16-bit TIF images and creates an MP4 video
with consistent brightness across all frames using global normalization.

PARAMETERS:
    input_dir (required)
        Path to directory containing TIF/TIFF images for video creation.
        Example: /path/to/tiff_images

    --output (optional, default: 500px_video.mp4)
        Output video file path.
        Example: --output my_video.mp4

    --width (optional, default: 500)
        Target video width in pixels. Height is calculated to maintain aspect ratio.
        Example: --width 800

    --fps (optional, default: 10)
        Video frame rate (frames per second).
        Example: --fps 15

    --crop (optional, default: None)
        Crop center region of each frame to specified box size in format WIDTHxHEIGHT.
        If not specified, entire image is used.
        Example: --crop 500x500

    --anchor (optional, default: False)
        Add a red dot anchor point at the center of each frame.
        Example: --anchor

    --global-normalize (optional, default: False)
        Use global min/max normalization for consistent brightness across all frames.
        By default, each frame is normalized independently (local normalization).
        Global normalization requires two passes through all images (slower but consistent).
        Example: --global-normalize

USAGE EXAMPLES:
    Basic usage with required input directory (local normalization):
        python create_video.py /path/to/tiff_images

    With custom resolution and fps:
        python create_video.py /path/to/tiff_images --width 800 --fps 15

    With center crop and anchor point:
        python create_video.py /path/to/tiff_images --crop 500x500 --anchor

    With global normalization (consistent brightness across all frames):
        python create_video.py /path/to/tiff_images --global-normalize

    Full customization:
        python create_video.py /path/to/tiff_images --output output.mp4 --width 1000 --fps 20 --crop 600x600 --anchor --global-normalize
"""

# Limit threading in numerical libraries to prevent CPU overload
# Each worker process should use only 1 thread
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import io
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import glob


def _process_image_minmax(image_path):
    """
    Worker function: Process a single image to find its min and max values.
    
    Args:
        image_path: Path to TIF image (string or Path object)
    
    Returns:
        tuple: (min_value, max_value)
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        return (float(img_array.min()), float(img_array.max()))
    except Exception as e:
        # Return infinite values if image cannot be processed
        # These will be filtered out when finding global min/max
        return (float('inf'), float('-inf'))


def find_global_min_max(image_paths, num_workers=1):
    """
    Scan all images to find global min and max pixel values using parallel processing.

    Args:
        image_paths: List of paths to TIF images
        num_workers: Number of parallel workers to use (default: 1)

    Returns:
        tuple: (global_min, global_max)
    """
    print("Pass 1: Finding global min/max values for normalization...")
    
    # Validate and use provided number of workers
    total_cores = mp.cpu_count()
    num_workers = max(1, min(num_workers, total_cores))  # Clamp between 1 and total_cores
    print(f"Using {num_workers} worker(s) out of {total_cores} available CPU cores")
    
    # Convert paths to strings for serialization
    work_items = [str(path) for path in image_paths]
    
    # Process in parallel with progress bar
    all_results = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for streaming results with progress tracking
        with tqdm(total=len(work_items), desc="Scanning images", unit="image", ncols=100) as pbar:
            for result in pool.imap_unordered(_process_image_minmax, work_items, chunksize=10):
                all_results.append(result)
                pbar.update(1)
    
    # Find global min/max from all results
    global_min = min(min_val for min_val, max_val in all_results if min_val != float('inf'))
    global_max = max(max_val for min_val, max_val in all_results if max_val != float('-inf'))
    
    print(f"Global min: {global_min}, Global max: {global_max}")
    return global_min, global_max


def get_local_min_max(img_array):
    """
    Get min and max pixel values from a single image.

    Args:
        img_array: Input image array

    Returns:
        tuple: (min_value, max_value)
    """
    return img_array.min(), img_array.max()


def crop_center(img_array, crop_width, crop_height):
    """
    Crop the center region of an image.

    Args:
        img_array: Input image array
        crop_width: Width of crop box
        crop_height: Height of crop box

    Returns:
        numpy.ndarray: Cropped image centered on the original
    """
    height, width = img_array.shape[:2]

    # Calculate crop coordinates to center the box
    start_x = max(0, (width - crop_width) // 2)
    start_y = max(0, (height - crop_height) // 2)
    end_x = min(width, start_x + crop_width)
    end_y = min(height, start_y + crop_height)

    return img_array[start_y:end_y, start_x:end_x]


def add_anchor_point(img_array, radius=5, color=(255, 0, 0)):
    """
    Add a red dot anchor point at the center of the image.

    Args:
        img_array: Input image array (grayscale or RGB)
        radius: Radius of the anchor dot in pixels
        color: RGB color tuple (default: red)

    Returns:
        numpy.ndarray: Image with anchor point (RGB format)
    """
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array.copy()

    # Calculate center coordinates
    height, width = img_rgb.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # Draw filled circle at center
    cv2.circle(img_rgb, (center_x, center_y), radius, color, -1)

    return img_rgb


def normalize_and_resize(img_array, norm_min, norm_max, target_width, crop_size=None):
    """
    Normalize 16-bit image to 8-bit, optionally crop, and resize.

    Args:
        img_array: Input 16-bit image array
        norm_min: Minimum pixel value for normalization
        norm_max: Maximum pixel value for normalization
        target_width: Target width in pixels
        crop_size: Optional tuple (width, height) for center cropping

    Returns:
        numpy.ndarray: Normalized, optionally cropped, and resized 8-bit image
    """
    # Apply center crop if specified
    if crop_size is not None:
        crop_width, crop_height = crop_size
        img_array = crop_center(img_array, crop_width, crop_height)

    # Normalize to 0-255 range using provided min/max
    if norm_max > norm_min:
        normalized = ((img_array - norm_min) / (norm_max - norm_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(img_array, dtype=np.uint8)

    # Resize maintaining aspect ratio
    height, width = normalized.shape
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized = cv2.resize(normalized, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return resized


def create_video(input_dir, output_path, target_width=500, fps=10, crop_size=None, show_anchor=False, use_global_normalize=False, num_workers=1):
    """
    Create video from TIF images with local or global normalization.

    Args:
        input_dir: Directory containing TIF images
        output_path: Output video file path
        target_width: Target video width in pixels
        fps: Frames per second
        crop_size: Optional tuple (width, height) for center cropping
        show_anchor: If True, add red dot at center of each frame
        use_global_normalize: If True, use global min/max for consistent brightness (2 passes)
        num_workers: Number of parallel workers for global normalization (default: 1)
    """
    input_dir = Path(input_dir)

    # Find all TIF files and sort by filename
    image_paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))

    if not image_paths:
        print(f"No TIF images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Estimated video duration: {len(image_paths) / fps:.1f} seconds ({len(image_paths) / fps / 60:.1f} minutes)")

    if crop_size:
        print(f"Center crop: {crop_size[0]}x{crop_size[1]} pixels")
    if show_anchor:
        print("Anchor point: Enabled (red dot at center)")

    # Determine normalization mode
    if use_global_normalize:
        print("Normalization: Global (consistent brightness across all frames)")
        # Pass 1: Find global min/max
        norm_min, norm_max = find_global_min_max(image_paths, num_workers)
    else:
        print("Normalization: Local (per-frame, faster processing)")
        # For local normalization, we'll calculate min/max per image
        # Set placeholder values here, will be calculated per-frame
        norm_min, norm_max = None, None

    # Get dimensions for video writer using first image
    first_img = Image.open(image_paths[0])
    first_array = np.array(first_img)

    # For dimension calculation, use local min/max if global not available
    if norm_min is None:
        first_min, first_max = get_local_min_max(first_array)
    else:
        first_min, first_max = norm_min, norm_max

    first_resized = normalize_and_resize(first_array, first_min, first_max, target_width, crop_size)

    # Apply anchor to determine final dimensions and color mode
    if show_anchor:
        first_with_anchor = add_anchor_point(first_resized)
        height, width = first_with_anchor.shape[:2]
        is_color = True
    else:
        height, width = first_resized.shape
        is_color = False

    print(f"\nOutput video resolution: {width}x{height}")
    print(f"Output file: {output_path}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=is_color)

    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return

    # Process images and write to video
    pass_label = "Pass 2: Creating video..." if use_global_normalize else "Creating video..."
    print(f"\n{pass_label}")
    for img_path in tqdm(image_paths, desc="Processing frames"):
        # Load 16-bit image
        img = Image.open(img_path)
        img_array = np.array(img)

        # Calculate normalization values (local or global)
        if use_global_normalize:
            frame_min, frame_max = norm_min, norm_max
        else:
            frame_min, frame_max = get_local_min_max(img_array)

        # Normalize and resize (with optional crop)
        frame = normalize_and_resize(img_array, frame_min, frame_max, target_width, crop_size)

        # Add anchor point if requested
        if show_anchor:
            frame = add_anchor_point(frame)

        # Write frame to video
        video_writer.write(frame)

    video_writer.release()
    print(f"\nVideo created successfully: {output_path}")
    print(f"Video info: {width}x{height}, {fps} fps, {len(image_paths)} frames")


def load_tiff_as_array(filepath):
    """Load a TIFF file and return it as a numpy array."""
    try:
        img = Image.open(filepath)
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def normalize_image(image_array, reference_array):
    """Normalize an image array pixel-wise using a reference array."""
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.divide(image_array, reference_array)
        normalized[reference_array == 0] = 0
    return normalized


def get_sorted_tiff_files(directory):
    """Get all TIFF files from directory sorted numerically."""
    directory = Path(directory)
    tiff_files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    
    def get_number(filepath):
        basename = filepath.stem
        # Try to extract number from filename
        try:
            return int(basename)
        except ValueError:
            # If not purely numeric, return string for alphabetical sort
            return basename
    
    return sorted(tiff_files, key=get_number)


def parse_time_from_filename(filename):
    """Parse time from filename format: HHMMSSMSEC (9 digits).
    
    Args:
        filename: e.g., '130329578.tif' or Path object
        
    Returns:
        Time in seconds (float), or None if parsing fails
    """
    try:
        if isinstance(filename, Path):
            basename = filename.stem
        else:
            basename = os.path.splitext(os.path.basename(filename))[0]
        
        if len(basename) != 9 or not basename.isdigit():
            return None
            
        hours = int(basename[0:2])
        minutes = int(basename[2:4])
        seconds = int(basename[4:6])
        milliseconds = int(basename[6:9])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
    except Exception as e:
        print(f"Error parsing time from {filename}: {e}")
        return None


def create_normalized_video_with_colormap(input_dir, output_path, fps=1, use_timestamps=False):
    """
    Create normalized video with BWR colormap, dynamic colorbar, and optional timestamps.
    
    Args:
        input_dir: Directory containing TIFF images
        output_path: Output video file path
        fps: Frames per second (default: 1)
        use_timestamps: If True, parse HHMMSSMSEC format and display timestamps
    
    Video specs:
        - Fixed size: 14x10 inches at 150 DPI = 2100x1500 pixels
        - Normalization: divide by first image (reference)
        - Color scaling: mean ± 3*std (dynamic per frame)
        - Colormap: BWR (blue-white-red)
        - High quality: libx264, CRF 18, 15 Mbps
    """
    print("=" * 60)
    print("Normalized Image Video Generator (with Colormap)")
    print("=" * 60)
    
    # Get all TIFF files
    print(f"\nScanning directory: {input_dir}")
    tiff_files = get_sorted_tiff_files(input_dir)
    
    if not tiff_files:
        print(f"Error: No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF images")
    
    # Load reference image (first image)
    ref_file = tiff_files[0]
    print(f"\nLoading reference image: {ref_file.name}")
    ref_array = load_tiff_as_array(ref_file)
    
    if ref_array is None:
        print("Error: Failed to load reference image")
        return
    
    print(f"Reference image shape: {ref_array.shape}")
    
    # Parse timestamps if requested
    time_deltas = None
    if use_timestamps:
        print("\nParsing timestamps...")
        first_time = parse_time_from_filename(tiff_files[0])
        
        if first_time is None:
            print(f"Error: Cannot parse timestamp from first file: {tiff_files[0].name}")
            print("Expected format: HHMMSSMSEC (9 digits)")
            return
        
        time_deltas = []
        for f in tiff_files:
            current_time = parse_time_from_filename(f)
            if current_time is None:
                print(f"Error: Cannot parse timestamp from {f.name}")
                print("Expected format: HHMMSSMSEC (9 digits)")
                return
            time_deltas.append(current_time - first_time)
        
        print(f"Time range: {time_deltas[0]:.3f}s to {time_deltas[-1]:.3f}s")
    
    # Load and normalize all images
    print("\nLoading and normalizing images...")
    normalized_images = []
    
    for i, f in enumerate(tiff_files):
        if i % 20 == 0 or i == len(tiff_files) - 1:
            print(f"  Processing image {i+1}/{len(tiff_files)}...")
        
        img_array = load_tiff_as_array(f)
        if img_array is None:
            print(f"Error: Failed to load {f.name}")
            return
        
        normalized = normalize_image(img_array, ref_array)
        normalized_images.append(normalized)
    
    print(f"Successfully normalized {len(normalized_images)} images")
    
    # Set up figure with GridSpec
    print("\nSetting up visualization...")
    fig = plt.figure(figsize=(14, 10))
    
    # Create GridSpec: image on left, colorbar on right
    gs = GridSpec(1, 2, figure=fig, width_ratios=[20, 1], wspace=0.15)
    
    ax_img = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    
    # Initial plot (will be updated in animation)
    initial_normalized = normalized_images[0]
    mean_val = np.mean(initial_normalized)
    std_val = np.std(initial_normalized)
    vmin = mean_val - 3 * std_val
    vmax = mean_val + 3 * std_val
    
    im = ax_img.imshow(initial_normalized, cmap='bwr', vmin=vmin, vmax=vmax)
    ax_img.set_xlabel('X Pixel', fontsize=12)
    ax_img.set_ylabel('Y Pixel', fontsize=12)
    
    # Initial colorbar
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Normalized Intensity', fontsize=12)
    
    # Title will be updated in animation
    if use_timestamps:
        title_text = f'Time: {time_deltas[0]:.3f}s'
    else:
        title_text = f'Frame: 1/{len(normalized_images)}'
    title = ax_img.set_title(title_text, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    # Animation update function
    def update_frame(frame_idx):
        """Update function for animation."""
        normalized = normalized_images[frame_idx]
        
        # Calculate dynamic color limits
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        min_val = np.min(normalized)
        max_val = np.max(normalized)
        
        vmin = max(mean_val - 3 * std_val, min_val)
        vmax = min(mean_val + 3 * std_val, max_val)
        
        # Update image and color limits
        im.set_data(normalized)
        im.set_clim(vmin=vmin, vmax=vmax)
        
        # Update title with timestamp or frame number
        if use_timestamps:
            title_text = f'Time: {time_deltas[frame_idx]:.3f}s'
        else:
            title_text = f'Frame: {frame_idx+1}/{len(normalized_images)}'
        title.set_text(title_text)
        
        # Update colorbar (need to redraw it)
        cbar.update_normal(im)
        
        if frame_idx % 20 == 0 or frame_idx == len(normalized_images) - 1:
            print(f"  Rendering frame {frame_idx+1}/{len(normalized_images)}...")
        
        return [im, title]
    
    # Create animation
    print(f"\nCreating animation at {fps} fps...")
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(normalized_images),
        interval=1000/fps, blit=False, repeat=False
    )
    
    # Save video with high quality settings
    print(f"Saving video to: {output_path}")
    print("This may take several minutes...")
    
    # FFmpeg writer with high quality settings
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(
            fps=fps,
            metadata=dict(artist='Normalized Image Video Generator'),
            bitrate=15000,  # High bitrate for quality: 15 Mbps
            codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18']
            # CRF 18 = visually lossless quality
        )
        
        anim.save(output_path, writer=writer, dpi=150)
        print("\n" + "=" * 60)
        print("SUCCESS! Normalized video created successfully!")
        print("=" * 60)
        print(f"\nOutput file: {output_path}")
        print(f"Number of frames: {len(normalized_images)}")
        print(f"Frame rate: {fps} fps")
        print(f"Duration: {len(normalized_images)/fps:.1f} seconds")
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"\nError saving video: {e}")
        print("\nTroubleshooting:")
        print("- Ensure ffmpeg is installed: sudo apt-get install ffmpeg")
        print("- Check disk space availability")
    finally:
        plt.close(fig)


class TextRedirector(io.StringIO):
    """Redirects stdout/stderr to a tkinter Text widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


class VideoCreatorGUI:
    """GUI for creating videos from TIF image sequences."""

    def __init__(self, root):
        self.root = root
        self.root.title("TIF to Video Creator")
        self.root.geometry("700x650")

        # Create main container with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Input directories (required) - multiple selection support
        ttk.Label(main_frame, text="Input Directories:*", font=("", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # Store list of directories
        self.input_directories = []

        # Directory list frame with listbox and scrollbar
        dir_list_frame = ttk.Frame(main_frame)
        dir_list_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        dir_list_frame.columnconfigure(0, weight=1)
        dir_list_frame.rowconfigure(0, weight=1)

        # Listbox with scrollbar
        dir_scrollbar = ttk.Scrollbar(dir_list_frame, orient=tk.VERTICAL)
        self.dir_listbox = tk.Listbox(dir_list_frame, height=4, yscrollcommand=dir_scrollbar.set)
        dir_scrollbar.config(command=self.dir_listbox.yview)
        self.dir_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        dir_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        row += 1

        # Buttons for managing directory list
        dir_button_frame = ttk.Frame(main_frame)
        dir_button_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        ttk.Button(dir_button_frame, text="Add Directory...", command=self.add_input_directory).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(dir_button_frame, text="Remove Selected", command=self.remove_selected_directory).grid(row=0, column=1)
        self.dir_count_label = ttk.Label(dir_button_frame, text="Selected directories: 0")
        self.dir_count_label.grid(row=0, column=2, padx=(15, 0))
        row += 1

        # Output files list (read-only display)
        ttk.Label(main_frame, text="Output Files:", font=("", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # Scrollable text widget for output file paths
        output_scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL)
        self.output_text = tk.Text(output_frame, height=4, wrap=tk.NONE, yscrollcommand=output_scrollbar.set, state='disabled')
        output_scrollbar.config(command=self.output_text.yview)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        row += 1

        # Width
        ttk.Label(main_frame, text="Video Width (pixels):", font=("", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        width_frame = ttk.Frame(main_frame)
        width_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.width_var = tk.IntVar(value=500)
        self.width_var.trace_add('write', lambda *args: self.update_output_filename())
        width_spinbox = ttk.Spinbox(width_frame, from_=100, to=4000, textvariable=self.width_var, width=10)
        width_spinbox.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(width_frame, text="(default: 500)").grid(row=0, column=1, sticky=tk.W, padx=5)
        row += 1

        # FPS
        ttk.Label(main_frame, text="Frame Rate (FPS):", font=("", 10)).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        fps_frame = ttk.Frame(main_frame)
        fps_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.fps_var = tk.IntVar(value=10)
        self.fps_var.trace_add('write', lambda *args: self.update_output_filename())
        fps_spinbox = ttk.Spinbox(fps_frame, from_=1, to=120, textvariable=self.fps_var, width=10)
        fps_spinbox.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(fps_frame, text="(default: 10)").grid(row=0, column=1, sticky=tk.W, padx=5)
        row += 1

        # Crop settings
        self.crop_enabled_var = tk.BooleanVar(value=False)
        crop_check = ttk.Checkbutton(main_frame, text="Enable Center Crop", variable=self.crop_enabled_var,
                                      command=self.toggle_crop)
        crop_check.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        crop_frame = ttk.Frame(main_frame)
        crop_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(crop_frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=(20, 5))
        self.crop_width_var = tk.IntVar(value=500)
        self.crop_width_var.trace_add('write', lambda *args: self.update_output_filename())
        self.crop_width_spinbox = ttk.Spinbox(crop_frame, from_=100, to=4000, textvariable=self.crop_width_var, width=10, state='disabled')
        self.crop_width_spinbox.grid(row=0, column=1, padx=5)

        ttk.Label(crop_frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.crop_height_var = tk.IntVar(value=500)
        self.crop_height_var.trace_add('write', lambda *args: self.update_output_filename())
        self.crop_height_spinbox = ttk.Spinbox(crop_frame, from_=100, to=4000, textvariable=self.crop_height_var, width=10, state='disabled')
        self.crop_height_spinbox.grid(row=0, column=3, padx=5)
        row += 1

        # Anchor point
        self.anchor_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Add red dot anchor at center", variable=self.anchor_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Global normalize
        self.global_normalize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Use global normalization (consistent brightness, slower)",
                        variable=self.global_normalize_var, command=self.toggle_global_normalize).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Workers control (for global normalization)
        workers_frame = ttk.Frame(main_frame)
        workers_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(workers_frame, text="Parallel workers:").grid(row=0, column=0, sticky=tk.W, padx=(20, 5))
        self.workers_var = tk.IntVar(value=1)
        self.workers_spinbox = ttk.Spinbox(workers_frame, from_=1, to=mp.cpu_count(), textvariable=self.workers_var, width=10, state='disabled')
        self.workers_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(workers_frame, text=f"(1 = safest for HDD, max = {mp.cpu_count()})").grid(row=0, column=2, sticky=tk.W, padx=5)
        row += 1

        # Create video button
        self.create_button = ttk.Button(main_frame, text="Create Video", command=self.start_video_creation)
        self.create_button.grid(row=row, column=0, columnspan=2, pady=15)
        row += 1

        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Normalized video section
        ttk.Label(main_frame, text="Normalized Video (with colormap):", font=("", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # Timestamp checkbox
        self.timestamp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Filenames have HHMMSSMSEC format (enables timestamp display)",
                        variable=self.timestamp_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Create normalized video button
        self.create_normalized_button = ttk.Button(main_frame, text="Create Normalized Video",
                                                    command=self.start_normalized_video_creation)
        self.create_normalized_button.grid(row=row, column=0, columnspan=2, pady=15)
        row += 1

        # Status/output area
        ttk.Label(main_frame, text="Status:", font=("", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(5, 0))
        row += 1

        self.status_text = scrolledtext.ScrolledText(main_frame, height=15, width=70, wrap=tk.WORD)
        self.status_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        main_frame.rowconfigure(row, weight=1)

        # Initial status message
        self.log_message("Ready to create videos. Please add input directories.\n")

        # Thread management
        self.processing_thread = None

    def toggle_crop(self):
        """Enable/disable crop spinboxes based on checkbox."""
        state = 'normal' if self.crop_enabled_var.get() else 'disabled'
        self.crop_width_spinbox.config(state=state)
        self.crop_height_spinbox.config(state=state)
        self.update_output_filename()

    def toggle_global_normalize(self):
        """Enable/disable workers spinbox based on global normalization checkbox."""
        state = 'normal' if self.global_normalize_var.get() else 'disabled'
        self.workers_spinbox.config(state=state)
        self.update_output_filename()

    def update_output_filename(self):
        """Update output files list based on current parameters and selected directories."""
        # Build filename based on parameters
        # Use try-except to handle cases where spinbox values are empty during initialization
        try:
            width = self.width_var.get()
            fps = self.fps_var.get()
        except (tk.TclError, ValueError):
            # Variables not yet initialized with valid values
            return

        filename_parts = [f"{width}px"]

        # Add FPS info
        filename_parts.append(f"{fps}fps")

        # Add crop info if enabled
        if self.crop_enabled_var.get():
            try:
                crop_w = self.crop_width_var.get()
                crop_h = self.crop_height_var.get()
            except (tk.TclError, ValueError):
                # Crop variables not yet initialized with valid values
                return
            filename_parts.append(f"crop_{crop_w}x{crop_h}")

        # Add global normalization flag if enabled
        if self.global_normalize_var.get():
            filename_parts.append("global_norm")

        filename_parts.append("video.mp4")
        filename = "_".join(filename_parts)

        # Update the output text widget with all output paths
        self.output_text.config(state='normal')  # Enable editing temporarily
        self.output_text.delete(1.0, tk.END)  # Clear current content

        if self.input_directories:
            # Generate and display all output paths
            for directory in self.input_directories:
                output_path = Path(directory) / filename
                self.output_text.insert(tk.END, str(output_path) + "\n")
        else:
            # Show message when no directories selected
            self.output_text.insert(tk.END, "No directories selected yet")

        self.output_text.config(state='disabled')  # Make read-only again
        self.output_text.see(1.0)  # Scroll to top

    def add_input_directory(self):
        """Add a directory to the input directory list."""
        directory = filedialog.askdirectory(title="Select Input Directory with TIF Images")
        if directory:
            # Check if directory already in list
            if directory in self.input_directories:
                messagebox.showinfo("Already Added", f"Directory already in the list:\n{directory}")
                return

            # Add to list
            self.input_directories.append(directory)
            self.dir_listbox.insert(tk.END, directory)

            # Update count label
            self.dir_count_label.config(text=f"Selected directories: {len(self.input_directories)}")

            self.log_message(f"Added directory: {directory}\n")

            # Update output filename pattern
            self.update_output_filename()

    def remove_selected_directory(self):
        """Remove selected directory from the input directory list."""
        selection = self.dir_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a directory to remove.")
            return

        # Get the index
        index = selection[0]
        directory = self.input_directories[index]

        # Remove from list and listbox
        self.input_directories.pop(index)
        self.dir_listbox.delete(index)

        # Update count label
        self.dir_count_label.config(text=f"Selected directories: {len(self.input_directories)}")

        self.log_message(f"Removed directory: {directory}\n")

        # Update output filename pattern
        self.update_output_filename()

    def log_message(self, message):
        """Add message to status text widget."""
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.update_idletasks()

    def validate_inputs(self):
        """Validate user inputs before starting video creation."""
        if not self.input_directories:
            messagebox.showerror("Error", "Please add at least one input directory.")
            return False

        # Validate all directories exist
        invalid_dirs = []
        for directory in self.input_directories:
            input_path = Path(directory)
            if not input_path.exists() or not input_path.is_dir():
                invalid_dirs.append(directory)

        if invalid_dirs:
            messagebox.showerror("Error",
                f"The following directories do not exist or are not valid:\n\n" +
                "\n".join(invalid_dirs))
            return False

        return True

    def start_video_creation(self):
        """Start video creation in a separate thread."""
        if not self.validate_inputs():
            return

        # Disable button during processing
        self.create_button.config(state='disabled', text="Processing...")
        self.status_text.delete(1.0, tk.END)
        self.log_message("Starting video creation...\n\n")

        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.create_video_thread, daemon=True)
        self.processing_thread.start()

    def create_video_thread(self):
        """Thread function to create videos for all directories without blocking GUI."""
        # Redirect stdout to GUI
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.status_text)

        total_dirs = len(self.input_directories)
        successful = 0
        failed = 0
        failed_dirs = []

        try:
            # Prepare crop size
            crop_size = None
            if self.crop_enabled_var.get():
                crop_size = (self.crop_width_var.get(), self.crop_height_var.get())

            # Get parameters
            target_width = self.width_var.get()
            fps = self.fps_var.get()
            show_anchor = self.anchor_var.get()
            use_global_normalize = self.global_normalize_var.get()
            num_workers = self.workers_var.get()

            # Build filename pattern
            width = target_width
            filename_parts = [f"{width}px", f"{fps}fps"]
            if crop_size:
                filename_parts.append(f"crop_{crop_size[0]}x{crop_size[1]}")
            if use_global_normalize:
                filename_parts.append("global_norm")
            filename_parts.append("video.mp4")
            filename = "_".join(filename_parts)

            # Process each directory
            for idx, input_dir in enumerate(self.input_directories, start=1):
                print(f"\n{'='*80}")
                print(f"Processing directory {idx} of {total_dirs}")
                print(f"Directory: {input_dir}")
                print(f"{'='*80}\n")

                try:
                    # Generate output path for this directory
                    output_path = Path(input_dir) / filename

                    # Call the create_video function
                    create_video(
                        input_dir=input_dir,
                        output_path=str(output_path),
                        target_width=target_width,
                        fps=fps,
                        crop_size=crop_size,
                        show_anchor=show_anchor,
                        use_global_normalize=use_global_normalize,
                        num_workers=num_workers
                    )

                    successful += 1
                    print(f"\n✓ Successfully completed directory {idx}/{total_dirs}\n")

                except Exception as e:
                    failed += 1
                    failed_dirs.append(input_dir)
                    error_msg = f"\n✗ Error processing directory {idx}/{total_dirs}: {str(e)}\n"
                    print(error_msg)

            # Show final summary
            print(f"\n{'='*80}")
            print(f"BATCH PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"Total directories: {total_dirs}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

            if failed_dirs:
                print(f"\nFailed directories:")
                for d in failed_dirs:
                    print(f"  - {d}")

            # Show success/failure message
            if failed == 0:
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    f"All videos created successfully!\n\nProcessed {successful} of {total_dirs} directories."))
            else:
                self.root.after(0, lambda: messagebox.showwarning("Partial Success",
                    f"Completed with some errors.\n\nSuccessful: {successful}/{total_dirs}\nFailed: {failed}/{total_dirs}\n\nCheck the status log for details."))

        except Exception as e:
            error_msg = f"\n\nFatal error: {str(e)}\n"
            print(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Fatal error during batch processing:\n{str(e)}"))

        finally:
            # Restore stdout
            sys.stdout = old_stdout

            # Re-enable button
            self.root.after(0, lambda: self.create_button.config(state='normal', text="Create Video"))

    def start_normalized_video_creation(self):
        """Start normalized video creation in a separate thread."""
        if not self.validate_inputs():
            return

        # Disable buttons during processing
        self.create_normalized_button.config(state='disabled', text="Processing...")
        self.create_button.config(state='disabled')
        self.status_text.delete(1.0, tk.END)
        self.log_message("Starting normalized video creation...\n\n")

        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.create_normalized_video_thread, daemon=True)
        self.processing_thread.start()

    def create_normalized_video_thread(self):
        """Thread function to create normalized videos for all directories without blocking GUI."""
        # Redirect stdout to GUI
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.status_text)

        total_dirs = len(self.input_directories)
        successful = 0
        failed = 0
        failed_dirs = []

        try:
            # Get parameters (only fps and timestamp for normalized video)
            fps = self.fps_var.get()
            use_timestamps = self.timestamp_var.get()

            # Build filename pattern for normalized video
            filename = f"normalized_{fps}fps_video.mp4"

            # Process each directory
            for idx, input_dir in enumerate(self.input_directories, start=1):
                print(f"\n{'='*80}")
                print(f"Processing directory {idx} of {total_dirs}")
                print(f"Directory: {input_dir}")
                print(f"{'='*80}\n")

                try:
                    # Generate output path for this directory
                    output_path = Path(input_dir) / filename

                    # Call the create_normalized_video_with_colormap function
                    create_normalized_video_with_colormap(
                        input_dir=input_dir,
                        output_path=str(output_path),
                        fps=fps,
                        use_timestamps=use_timestamps
                    )

                    successful += 1
                    print(f"\n✓ Successfully completed directory {idx}/{total_dirs}\n")

                except Exception as e:
                    failed += 1
                    failed_dirs.append(input_dir)
                    error_msg = f"\n✗ Error processing directory {idx}/{total_dirs}: {str(e)}\n"
                    print(error_msg)

            # Show final summary
            print(f"\n{'='*80}")
            print(f"BATCH PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"Total directories: {total_dirs}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")

            if failed_dirs:
                print(f"\nFailed directories:")
                for d in failed_dirs:
                    print(f"  - {d}")

            # Show success/failure message
            if failed == 0:
                self.root.after(0, lambda: messagebox.showinfo("Success",
                    f"All normalized videos created successfully!\n\nProcessed {successful} of {total_dirs} directories."))
            else:
                self.root.after(0, lambda: messagebox.showwarning("Partial Success",
                    f"Completed with some errors.\n\nSuccessful: {successful}/{total_dirs}\nFailed: {failed}/{total_dirs}\n\nCheck the status log for details."))

        except Exception as e:
            error_msg = f"\n\nFatal error: {str(e)}\n"
            print(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Fatal error during batch processing:\n{str(e)}"))

        finally:
            # Restore stdout
            sys.stdout = old_stdout

            # Re-enable buttons
            self.root.after(0, lambda: self.create_normalized_button.config(state='normal', text="Create Normalized Video"))
            self.root.after(0, lambda: self.create_button.config(state='normal'))


def main():
    parser = argparse.ArgumentParser(
        description="Create video from TIF image sequence with local or global normalization (GUI mode by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  GUI mode (default):
    %(prog)s

  Command-line mode:
    %(prog)s --nogui /path/to/tiff_images
    %(prog)s --nogui /path/to/tiff_images --width 800 --fps 15
    %(prog)s --nogui /path/to/tiff_images --crop 500x500 --anchor
    %(prog)s --nogui /path/to/tiff_images --global-normalize
    %(prog)s --nogui /path/to/tiff_images --output output.mp4 --width 1000 --fps 20 --crop 600x600 --anchor --global-normalize
        """
    )

    # Optional positional argument for CLI mode
    parser.add_argument(
        "input_dir",
        type=str,
        nargs='?',
        default=None,
        help="Input directory containing TIF images (required for --nogui mode)"
    )

    # GUI/CLI mode flag
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run in command-line mode instead of GUI mode (requires input_dir)"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default="500px_video.mp4",
        help="Output video file path (default: 500px_video.mp4)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=500,
        help="Target video width in pixels (default: 500)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop center region to specified box size in format WIDTHxHEIGHT (e.g., 500x500)"
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="Add red dot anchor point at the center of each frame"
    )
    parser.add_argument(
        "--global-normalize",
        action="store_true",
        help="Use global min/max normalization for consistent brightness across all frames (requires 2 passes, slower)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for global normalization (default: 1, recommended for HDD)"
    )

    args = parser.parse_args()

    # Determine mode: GUI or CLI
    if args.nogui:
        # CLI mode - require input_dir
        if args.input_dir is None:
            parser.error("input_dir is required when using --nogui mode")

        # Parse crop size if provided
        crop_size = None
        if args.crop:
            try:
                crop_parts = args.crop.lower().split('x')
                if len(crop_parts) != 2:
                    raise ValueError("Crop format must be WIDTHxHEIGHT (e.g., 500x500)")
                crop_width = int(crop_parts[0])
                crop_height = int(crop_parts[1])
                crop_size = (crop_width, crop_height)
            except (ValueError, IndexError):
                parser.error(f"Invalid crop format: {args.crop}. Use format WIDTHxHEIGHT (e.g., 500x500)")

        # Run in CLI mode
        create_video(
            input_dir=args.input_dir,
            output_path=args.output,
            target_width=args.width,
            fps=args.fps,
            crop_size=crop_size,
            show_anchor=args.anchor,
            use_global_normalize=args.global_normalize,
            num_workers=args.workers
        )
    else:
        # GUI mode (default)
        root = tk.Tk()
        app = VideoCreatorGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
