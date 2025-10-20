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

    --workers (optional, default: CPU count - 1)
        Number of parallel worker processes for faster processing.
        Uses multiprocessing to parallelize image loading and processing.
        Default uses all available CPU cores minus one.
        Example: --workers 4

USAGE EXAMPLES:
    Basic usage with required input directory (local normalization):
        python create_video.py /path/to/tiff_images

    With custom resolution and fps:
        python create_video.py /path/to/tiff_images --width 800 --fps 15

    With center crop and anchor point:
        python create_video.py /path/to/tiff_images --crop 500x500 --anchor

    With global normalization (consistent brightness across all frames):
        python create_video.py /path/to/tiff_images --global-normalize

    With parallel processing (faster on multi-core systems):
        python create_video.py /path/to/tiff_images --workers 8

    Full customization:
        python create_video.py /path/to/tiff_images --output output.mp4 --width 1000 --fps 20 --crop 600x600 --anchor --global-normalize --workers 8
"""

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
from multiprocessing import Pool, Queue, Process
import time
from functools import partial


# ============================================================================
# Multiprocessing Helper Functions
# ============================================================================

def _process_image_min_max(img_path):
    """
    Worker function to find min/max values in a single image.

    Args:
        img_path: Path to TIF image

    Returns:
        tuple: (min_value, max_value)
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        return (img_array.min(), img_array.max())
    except Exception as e:
        print(f"Warning: Error processing {img_path}: {e}")
        return (np.inf, -np.inf)


def _process_frame_worker(img_path, norm_min, norm_max, target_width, crop_size):
    """
    Worker function to load and preprocess a single frame.

    Args:
        img_path: Path to TIF image
        norm_min: Minimum pixel value for normalization
        norm_max: Maximum pixel value for normalization
        target_width: Target width in pixels
        crop_size: Optional tuple (width, height) for center cropping

    Returns:
        numpy.ndarray: Processed frame (8-bit, resized)
    """
    try:
        # Load 16-bit image
        img = Image.open(img_path)
        img_array = np.array(img)

        # If using local normalization, calculate per-frame min/max
        if norm_min is None:
            frame_min, frame_max = img_array.min(), img_array.max()
        else:
            frame_min, frame_max = norm_min, norm_max

        # Normalize and resize (with optional crop)
        frame = normalize_and_resize(img_array, frame_min, frame_max, target_width, crop_size)

        return frame
    except Exception as e:
        print(f"Warning: Error processing frame {img_path}: {e}")
        return None


# ============================================================================
# Original Functions (Updated for Parallel Processing)
# ============================================================================

def find_global_min_max(image_paths, num_workers=None):
    """
    Scan all images to find global min and max pixel values using parallel processing.

    Args:
        image_paths: List of paths to TIF images
        num_workers: Number of worker processes (default: CPU count - 1)

    Returns:
        tuple: (global_min, global_max)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"Pass 1: Finding global min/max values for normalization... (using {num_workers} workers)")

    # Use multiprocessing pool to process images in parallel
    with Pool(processes=num_workers) as pool:
        # Map the worker function across all image paths with progress bar
        results = list(tqdm(
            pool.imap(_process_image_min_max, image_paths),
            total=len(image_paths),
            desc="Scanning images"
        ))

    # Reduce results to find global min/max
    global_min = min(r[0] for r in results)
    global_max = max(r[1] for r in results)

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


def create_video(input_dir, output_path, target_width=500, fps=10, crop_size=None, show_anchor=False, use_global_normalize=False, num_workers=None):
    """
    Create video from TIF images with local or global normalization using parallel processing.

    Args:
        input_dir: Directory containing TIF images
        output_path: Output video file path
        target_width: Target video width in pixels
        fps: Frames per second
        crop_size: Optional tuple (width, height) for center cropping
        show_anchor: If True, add red dot at center of each frame
        use_global_normalize: If True, use global min/max for consistent brightness (2 passes)
        num_workers: Number of worker processes (default: CPU count - 1)
    """
    start_time = time.time()

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
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
        norm_min, norm_max = find_global_min_max(image_paths, num_workers=num_workers)
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

    # Process images and write to video using parallel processing
    pass_label = "Pass 2: Creating video..." if use_global_normalize else "Creating video..."
    print(f"\n{pass_label} (using {num_workers} workers)")

    # Create partial function with fixed parameters
    process_func = partial(
        _process_frame_worker,
        norm_min=norm_min,
        norm_max=norm_max,
        target_width=target_width,
        crop_size=crop_size
    )

    # Process frames in parallel using multiprocessing pool
    with Pool(processes=num_workers) as pool:
        # Use imap to maintain order and process with progress bar
        for frame in tqdm(
            pool.imap(process_func, image_paths),
            total=len(image_paths),
            desc="Processing frames"
        ):
            if frame is not None:
                # Add anchor point if requested (must be done sequentially)
                if show_anchor:
                    frame = add_anchor_point(frame)

                # Write frame to video (must be done sequentially)
                video_writer.write(frame)

    video_writer.release()

    # Calculate and display performance metrics
    elapsed_time = time.time() - start_time
    frames_per_second = len(image_paths) / elapsed_time if elapsed_time > 0 else 0

    print(f"\nVideo created successfully: {output_path}")
    print(f"Video info: {width}x{height}, {fps} fps, {len(image_paths)} frames")
    print(f"Processing time: {elapsed_time:.2f} seconds ({frames_per_second:.2f} frames/sec)")
    print(f"Workers used: {num_workers}")


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
                        variable=self.global_normalize_var, command=self.update_output_filename).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        # Create video button
        self.create_button = ttk.Button(main_frame, text="Create Video", command=self.start_video_creation)
        self.create_button.grid(row=row, column=0, columnspan=2, pady=15)
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

                    # Call the create_video function with multiprocessing enabled
                    create_video(
                        input_dir=input_dir,
                        output_path=str(output_path),
                        target_width=target_width,
                        fps=fps,
                        crop_size=crop_size,
                        show_anchor=show_anchor,
                        use_global_normalize=use_global_normalize,
                        num_workers=None  # Use default (CPU count - 1)
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
        default=None,
        help=f"Number of parallel worker processes (default: {max(1, mp.cpu_count() - 1)} = CPU count - 1)"
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
    # Required for multiprocessing on Windows and macOS
    mp.freeze_support()
    main()
