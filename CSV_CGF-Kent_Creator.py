#!/usr/bin/env python3
# CSV_CGF-Kent_Creator.py - Version 0.23
# High-performance script with binary search for sequence detection, optimized scanning, and enhanced UI

import os
import csv
import sys
import re
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import traceback
from typing import List, Dict, Any, Optional, Set, Tuple
import concurrent.futures
import argparse
import logging
import mimetypes
from collections import defaultdict

# Determine if application is a script file or frozen exe
FROZEN = getattr(sys, 'frozen', False)

# Set up output capture for diagnostics when running as executable
if FROZEN:
    # Redirect stdout and stderr to files when running as bundled application
    sys.stdout = open(os.path.join(os.path.expanduser("~"), 'csv_creator_stdout.log'), 'w')
    sys.stderr = open(os.path.join(os.path.expanduser("~"), 'csv_creator_stderr.log'), 'w')

# Try to import these, but don't fail if they're not available (for CLI mode)
GUI_AVAILABLE = False
try:
    import cv2  # For video duration
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QPushButton, QLabel,
        QTableWidget, QTableWidgetItem,
        QVBoxLayout, QHBoxLayout, QProgressBar, QWidget,
        QLineEdit, QGroupBox, QCheckBox, QMessageBox,
        QListWidget, QListWidgetItem, QFileDialog,
        QPlainTextEdit, QGridLayout, QSpinBox, QComboBox,
        QTabWidget, QSplitter, QMenu, QAction, QErrorMessage,
        QDialog, QHeaderView, QInputDialog
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QSize, QPoint, QTimer
    from PyQt5.QtGui import QIcon, QFont, QCursor
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"GUI libraries not available: {e}")
    if FROZEN:
        # This is critical if running as executable - log the error
        with open(os.path.join(os.path.expanduser("~"), 'csv_creator_import_error.log'), 'w') as f:
            f.write(f"Import error: {e}\n")
            f.write(traceback.format_exc())

# Set up logging
log_file = os.path.join(os.path.expanduser("~"), f"file_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('CSV_Creator')

# --------------------
# Constants and Default Settings
# --------------------
VERSION = "0.23"
DEFAULT_SETTINGS = {
    "sequence_min": 3,
    "sequence_max": 4,
    "use_shot": True,
    "use_pixel": True,
    "use_fps": True,
    "version_min": 3,
    "version_max": 3,
    "sequence_threshold": 5,
    "vendor_name": "CG Fluids",
    "upload_status": "Uploaded to Aspera",
    "last_delivery_package": "",
    "image_types": ["exr", "jpg", "jpeg", "tiff", "tif", "png", "tga", "psd"],
    "video_types": ["mov", "mxf", "mp4", "avi", "webm", "wmv"],
    "other_types": ["zip", "tar", "gz", "rar", "7z", "pdf", "txt", "csv", "json", "xml", "html"],
    "scan_other_types": True,
    "recursive_scan": True,  
    "safe_mode": False,       # Changed default to False for better performance
    "worker_threads": 8,
    "file_size_warning": 1000,  # Increased to 1GB to avoid unnecessary safe handling
    "window_geometry": None,
    "last_directories": [],
    "binary_search": False,    # Disabled by default since it may cause detection issues
    "max_sampling_depth": 10,
    "custom_submitted_for": ["Previz", "Rnd", "Look Dev", "Final Pend tech", "Final Tech checked"],
    "custom_delivery_notes": []
}

# Initialize mimetypes
mimetypes.init()

# --------------------
# Dark Theme Style Sheet
# --------------------
DARK_STYLESHEET = """
QWidget { background-color: #2d2d30; color: #ffffff; font-family: Arial; }
QPushButton { 
    background-color: #3e3e42; 
    border: 1px solid #565656; 
    padding: 5px; 
    border-radius: 3px;
}
QPushButton:hover { background-color: #46464b; }
QPushButton:pressed { background-color: #007acc; }
QLineEdit, QTableWidget, QListWidget, QPlainTextEdit, QSpinBox, QComboBox {
    background-color: #3e3e42; color: #ffffff; border: 1px solid #565656;
    selection-background-color: #007acc;
}
QGroupBox { 
    border: 1px solid #565656; 
    margin-top: 10px; 
    border-radius: 3px;
}
QGroupBox::title {
    subcontrol-origin: margin; 
    subcontrol-position: top center; 
    padding: 0 5px;
    background-color: #2d2d30;
}
QCheckBox { spacing: 5px; }
QProgressBar { 
    border: 1px solid #565656; 
    text-align: center; 
    background-color: #3e3e42; 
    border-radius: 3px;
}
QProgressBar::chunk { background-color: #007acc; }
QHeaderView::section { 
    background-color: #3e3e42; 
    padding: 4px; 
    border: 1px solid #565656; 
}
QTabWidget::pane { border: 1px solid #565656; }
QTabBar::tab { 
    background-color: #3e3e42; 
    padding: 6px 10px; 
    border: 1px solid #565656; 
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected { 
    background-color: #007acc; 
}
QTabBar::tab:!selected { 
    margin-top: 2px; 
}
QMenu {
    background-color: #2d2d30;
    border: 1px solid #565656;
}
QMenu::item {
    padding: 5px 20px 5px 20px;
}
QMenu::item:selected {
    background-color: #007acc;
}
QSplitter::handle {
    background-color: #565656;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #3e3e42;
    width: 10px;
    height: 10px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #565656;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #007acc;
}
"""

# Get application resource path (works both in development and bundled mode)
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def build_pattern(seq_min, seq_max, use_shot, use_pixel, use_fps, ver_min, ver_max):
    """
    Build a regex pattern for filename validation based on given parameters.
    Updated for v0.17+ to match the correct pattern format.
    """
    parts = []
    # Sequence and shot without underscore between them
    parts.append(r'^(?P<sequence>[A-Za-z]{%d,%d})' % (seq_min, seq_max))
    if use_shot:
        parts.append(r'(?:(?P<shotNumber>\d{4}))?')
    parts.append(r'_(?P<description>[\w-]+)_')
    
    # Pixel mapping and resolution without underscore between them
    if use_pixel:
        parts.append(r'(?:(?P<pixelMapping>LL180|LL360))?')
    parts.append(r'(?P<resolution>\d+[kK])_')  # Note: K can be uppercase or lowercase
    
    parts.append(r'(?P<colorspaceGamma>[^_]+)')
    if use_fps:
        parts.append(r'(?:_(?P<fps>\d+))?_')
    else:
        parts.append(r'_')
    parts.append(r'v(?P<version>\d{%d,%d})' % (ver_min, ver_max))
    parts.append(r'(?:[_\.](?P<frame_padding>\d+)\.(?P<extension>[^.]+)'
                 r'|\.(?P<extension2>[^.]+))$')
    return re.compile(''.join(parts), re.IGNORECASE)

class ThreadSafeTimeout:
    """A thread-safe timeout implementation that doesn't rely on signals"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None
        self.timed_out = False
        
    def __enter__(self):
        def timeout_handler():
            self.timed_out = True
        
        self.timer = threading.Timer(self.seconds, timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        
        if self.timed_out:
            raise TimeoutError(f"Operation timed out after {self.seconds} seconds")
        return False

class Settings:
    """
    Class to handle application settings across sessions and modes.
    """
    def __init__(self, settings_file=None):
        self.settings_file = settings_file or os.path.join(
            os.path.expanduser("~"), 
            ".cg_fluids_scanner_settings.json"
        )
        self.data = DEFAULT_SETTINGS.copy()
        self.load()
    
    def load(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    # Only update keys that already exist to avoid errors with old settings files
                    for key in self.data:
                        if key in loaded:
                            self.data[key] = loaded[key]
                    
                    # Handle new keys that might not exist in old settings files
                    if "other_types" not in loaded:
                        self.data["other_types"] = DEFAULT_SETTINGS["other_types"]
                    if "scan_other_types" not in loaded:
                        self.data["scan_other_types"] = DEFAULT_SETTINGS["scan_other_types"]
                    if "recursive_scan" not in loaded:
                        self.data["recursive_scan"] = DEFAULT_SETTINGS["recursive_scan"]
                    if "safe_mode" not in loaded:
                        self.data["safe_mode"] = DEFAULT_SETTINGS["safe_mode"]
                    if "worker_threads" not in loaded:
                        self.data["worker_threads"] = DEFAULT_SETTINGS["worker_threads"]
                    if "file_size_warning" not in loaded:
                        self.data["file_size_warning"] = DEFAULT_SETTINGS["file_size_warning"]
                    if "binary_search" not in loaded:
                        self.data["binary_search"] = DEFAULT_SETTINGS["binary_search"]
                    if "max_sampling_depth" not in loaded:
                        self.data["max_sampling_depth"] = DEFAULT_SETTINGS["max_sampling_depth"]
                    if "custom_submitted_for" not in loaded:
                        self.data["custom_submitted_for"] = DEFAULT_SETTINGS["custom_submitted_for"]
                    if "custom_delivery_notes" not in loaded:
                        self.data["custom_delivery_notes"] = DEFAULT_SETTINGS["custom_delivery_notes"]
                        
                logger.info(f"Settings loaded from {self.settings_file}")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def save(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
        
    def update(self, data_dict):
        self.data.update(data_dict)
        self.save()

class FileUtils:
    """
    Utility functions for handling files
    """
    # Cache for file sizes to avoid repeated disk access
    _file_size_cache = {}
    
    @staticmethod
    def get_file_size_bytes(file_path):
        """Get file size in bytes, with error handling and caching"""
        try:
            # Check cache first
            if file_path in FileUtils._file_size_cache:
                return FileUtils._file_size_cache[file_path]
                
            size = os.path.getsize(file_path)
            # Cache the result
            FileUtils._file_size_cache[file_path] = size
            return size
        except Exception as e:
            # logger.error(f"Error getting file size for {file_path}: {e}")
            return 0

    @staticmethod
    def get_file_size(file_path):
        """Get file size in human-readable format, with error handling"""
        try:
            size_bytes = FileUtils.get_file_size_bytes(file_path)
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} PB"
        except Exception as e:
            # logger.error(f"Error formatting file size for {file_path}: {e}")
            return "Unknown"
    
    @staticmethod
    def is_large_file(file_path, threshold_mb=500):
        """Check if file is larger than threshold (default 500MB)"""
        try:
            size_bytes = FileUtils.get_file_size_bytes(file_path)
            return size_bytes > (threshold_mb * 1024 * 1024)
        except Exception:
            # If we can't check the size, assume it's not large
            return False
    
    @staticmethod
    def get_file_type(file_path):
        """Get file type by extension and MIME type"""
        try:
            ext = os.path.splitext(file_path)[1].lower()[1:]
            mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type:
                main_type = mime_type.split('/')[0]
                if main_type == 'image':
                    return 'Image'
                elif main_type == 'video':
                    return 'Video'
                elif main_type == 'audio':
                    return 'Audio'
                elif main_type == 'application':
                    if 'zip' in mime_type or 'compressed' in mime_type:
                        return 'Archive'
                    if 'pdf' in mime_type:
                        return 'Document'
                elif main_type == 'text':
                    return 'Text'
            
            # Fallback to extension-based detection
            return ext.upper() if ext else "Unknown"
        except Exception as e:
            # logger.error(f"Error determining file type for {file_path}: {e}")
            return "Unknown"
    
    @staticmethod
    def extension_matches(filename, extensions):
        """Check if filename has any of the given extensions"""
        if not extensions:
            return False
            
        ext = os.path.splitext(filename)[1].lower()[1:]
        return ext.lower() in [e.lower() for e in extensions]

    @staticmethod
    def safe_read_file(file_path, timeout=10):
        """
        Safely read a file with timeout to prevent hanging.
        Uses a thread-safe timeout implementation that works in all threads.
        """
        try:
            with ThreadSafeTimeout(timeout):
                with open(file_path, 'rb') as f:
                    # Just read a small amount to check accessibility
                    f.read(1024)
                return True
        except Exception as e:
            # logger.warning(f"Safe read test failed for {file_path}: {e}")
            return False

    @staticmethod
    def clear_caches():
        """Clear all internal caches"""
        FileUtils._file_size_cache.clear()

class FileProcessor:
    """
    Core functionality for file processing, can be used by GUI or CLI.
    """
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.pattern = None
        self.update_pattern()
        self.dir_cache = {}
        self.file_cache = {}
        self.fast_validation_cache = {}  # For quick validation checks
        self.sequence_cache = {}         # Cache for sequence detection
        self.image_types = self.settings.get("image_types")
        self.video_types = self.settings.get("video_types")
        self.other_types = self.settings.get("other_types")
        self.safe_mode = self.settings.get("safe_mode", False)  # Changed default to False for better performance
        self.file_size_warning = self.settings.get("file_size_warning", 1000)  # Increased threshold to 1GB
        self.binary_search = self.settings.get("binary_search", True)
        self.max_sampling_depth = self.settings.get("max_sampling_depth", 10)
        
    def update_pattern(self):
        self.pattern = build_pattern(
            self.settings.get("sequence_min"),
            self.settings.get("sequence_max"),
            self.settings.get("use_shot"),
            self.settings.get("use_pixel"),
            self.settings.get("use_fps"),
            self.settings.get("version_min"),
            self.settings.get("version_max")
        )
    def scan_directories(self, directories, file_types, delivery_package, 
                progress_callback=None, log_callback=None, preview_callback=None,
                include_other_types=False):
        """
        Scan directories for files matching the pattern and generate report data.
        Can be used by CLI or GUI mode with appropriate callbacks.
        
        Optimized version with parallel scanning, improved binary search, and faster processing.
        """
        def emit_log(msg):
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        
        def emit_progress(value):
            if progress_callback:
                progress_callback(value)
        
        def emit_preview(data):
            if preview_callback:
                preview_callback(data)
        
        # Clear caches at the start of a new scan
        self.file_cache.clear()
        self.dir_cache.clear()
        self.fast_validation_cache.clear()
        self.sequence_cache.clear()
        FileUtils.clear_caches()
        
        # Fixed function to correctly traverse directories and subdirectories
        def walker(dirs):
            """
            Recursively walk through directories to find files.
            Highly optimized implementation with parallel directory scanning and early pattern detection.
            """
            all_files = []
            sequence_patterns = set()  # Store unique sequence patterns for binary search
            processed_dirs = 0
            total_dirs = len(dirs)
            
            # Precompile extension checkers for performance
            image_exts = set('.' + ext.lower() for ext in self.image_types)
            video_exts = set('.' + ext.lower() for ext in self.video_types)
            other_exts = set('.' + ext.lower() for ext in self.other_types) if include_other_types else set()
            all_exts = image_exts | video_exts | other_exts
            
            # Precompile sequence pattern regex for fast detection - optimized for speed
            seq_pattern = re.compile(r'^([A-Za-z]{1,5})(?:\d{4})?_[\w-]+_(?:(?:LL180|LL360))?(?:\d+[kK])_[^_]+(?:_\d+)?_v\d+(?:[_.](\d+)\.([^.]+)|\.([^.]+))$', re.IGNORECASE)
            
            # Create a thread pool for parallel directory scanning
            # Number of workers is limited to avoid excessive thread creation
            scan_workers = min(self.settings.get("worker_threads", 8), len(dirs), os.cpu_count() or 4)
            
            # Track already visited directories to avoid duplicates
            visited_dirs = set()
            
            # Process a single directory
            def process_directory(dir_path):
                dir_files = []
                dir_sequences = set()
                
                # Check if we should recursively scan
                if self.settings.get("recursive_scan", True):
                    for root, _, files in os.walk(dir_path):
                        if root in visited_dirs:
                            continue
                            
                        visited_dirs.add(root)
                        
                        for filename in files:
                            # Fast extension check
                            ext_pos = filename.rfind('.')
                            if ext_pos == -1:
                                continue
                                
                            ext = filename[ext_pos:].lower()
                            if ext in all_exts:
                                file_path = os.path.join(root, filename)
                                dir_files.append(file_path)
                                
                                # Check for sequence pattern - used for binary search
                                if ext in image_exts and seq_pattern.match(filename):
                                    m = seq_pattern.match(filename)
                                    if m and m.group(2) and m.group(3):  # Has frame number and extension
                                        frame_num = m.group(2)
                                        ext_val = m.group(3)
                                        # Get base pattern by removing frame number
                                        base_pattern = re.sub(r'[_.]' + re.escape(frame_num) + r'\.' + re.escape(ext_val) + r'$', '', filename)
                                        dir_sequences.add((root, base_pattern, ext_val))
                else:
                    # Non-recursive mode: only scan the top level of the directory
                    try:
                        for entry in os.scandir(dir_path):
                            if entry.is_file():
                                # Fast extension check
                                name = entry.name
                                ext_pos = name.rfind('.')
                                if ext_pos == -1:
                                    continue
                                    
                                ext = name[ext_pos:].lower()
                                if ext in all_exts:
                                    file_path = entry.path
                                    dir_files.append(file_path)
                                    
                                    # Check for sequence pattern
                                    if ext in image_exts and seq_pattern.match(name):
                                        m = seq_pattern.match(name)
                                        if m and m.group(2) and m.group(3):  # Has frame number and extension
                                            frame_num = m.group(2)
                                            ext_val = m.group(3)
                                            # Get base pattern by removing frame number
                                            base_pattern = re.sub(r'[_.]' + re.escape(frame_num) + r'\.' + re.escape(ext_val) + r'$', '', name)
                                            dir_sequences.add((dir_path, base_pattern, ext_val))
                    except Exception as e:
                        emit_log(f"Error scanning directory {dir_path}: {e}")
                        
                return dir_files, dir_sequences
            
            # Process directories in parallel
            if scan_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers) as executor:
                    futures = [executor.submit(process_directory, d) for d in dirs]
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        dir_files, dir_sequences = future.result()
                        all_files.extend(dir_files)
                        sequence_patterns.update(dir_sequences)
                        
                        processed_dirs += 1
                        if dir_files:
                            emit_log(f"Found {len(dir_files)} matching files in directory {processed_dirs}/{total_dirs}")
                        emit_progress(int(10 * processed_dirs / total_dirs))  # Update progress during directory scan
            else:
                # Single-threaded version for small directory sets
                for d in dirs:
                    dir_files, dir_sequences = process_directory(d)
                    all_files.extend(dir_files)
                    sequence_patterns.update(dir_sequences)
                    
                    processed_dirs += 1
                    if dir_files:
                        emit_log(f"Found {len(dir_files)} matching files in directory {processed_dirs}/{total_dirs}")
                    emit_progress(int(10 * processed_dirs / total_dirs))
            
            # Log the final count
            if all_files:
                emit_log(f"Total files found: {len(all_files)}")
                # Log some sample file paths to verify
                max_samples = min(5, len(all_files))
                emit_log(f"Sample files (showing {max_samples} of {len(all_files)}):")
                for i in range(max_samples):
                    emit_log(f"  - {all_files[i]}")
                
                # Log sequence patterns if binary search is enabled
                if self.binary_search and sequence_patterns:
                    emit_log(f"Found {len(sequence_patterns)} potential image sequences for binary search")
            else:
                emit_log("No matching files found. Check file types and directory paths.")
                # Log selected file types for debugging
                emit_log(f"Selected file types: {file_types}")
                if include_other_types:
                    emit_log(f"Also including other file types: {self.other_types}")
            
            return all_files, sequence_patterns
            
        # Send headers to preview callback
        headers = [
            "Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes",
            "File Type", "Resolution", "Duration/Size", "Delivery Date",
            "Delivery Package Name", "Upload Status", "Vendor Name"
        ]
        emit_preview({"action": "init", "headers": headers})
        
        # Find all matching files
        emit_log(f"Scanning directories: {directories}")
        emit_log(f"Recursive scan: {self.settings.get('recursive_scan', True)}")
        emit_log(f"Safe mode: {self.settings.get('safe_mode', True)}")
        emit_log(f"Binary search: {self.settings.get('binary_search', True)}")
        
        time_start = time.time()
        files, sequence_patterns = walker(directories)
        total = len(files)
        emit_log(f"Found {total} candidate files in {time.time() - time_start:.2f} seconds")
        
        if total == 0:
            emit_log("No files to process")
            emit_preview({"action": "complete"})
            return headers, []
        
        # Process files in parallel using a thread pool with improved batching
        emit_progress(10)  # Show some initial progress after walker completes
        
        # Define an optimized batch processing function with early validation
        def process_batch(file_batch):
            """Process a batch of files with optimized validation"""
            batch_results = []
            
            # Create local caches to avoid thread synchronization overhead
            local_validation_cache = {}
            local_file_size_cache = {}
            
            for file_path in file_batch:
                # Quick extension check before processing
                ext = os.path.splitext(file_path)[1].lower()[1:] if '.' in file_path else ''
                
                # Skip files with invalid extensions quickly
                if not (ext in self.image_types or ext in self.video_types or 
                        (include_other_types and ext in self.other_types)):
                    continue
                
                # Skip files that don't exist or are inaccessible
                if not os.path.exists(file_path):
                    continue
                    
                # Process the file
                result = self.process_file(file_path)
                if result['valid']:
                    batch_results.append(result['data'])
                    
            return batch_results
        
        # Optimize batch size based on file count
        total_files = len(files)
        batch_size = 100  # Default batch size
        
        # Adjust batch size based on total file count for better performance
        if total_files > 10000:
            batch_size = 200
        elif total_files > 50000:
            batch_size = 500
        elif total_files < 1000:
            batch_size = 50
        
        # Create batches with optimized size
        batches = [files[i:i+batch_size] for i in range(0, total_files, batch_size)]
        total_batches = len(batches)
        
        # Determine optimal worker count - don't create too many threads for small jobs
        worker_count = min(
            self.settings.get("worker_threads", 8),  # User setting
            os.cpu_count() or 4,                     # CPU core count
            max(1, min(16, total_batches // 2))      # Dynamic based on batch count
        )
        
        emit_log(f"Processing {total_files} files in {total_batches} batches using {worker_count} worker threads")
        
        # Process files with progress updates
        processed_files = 0
        parsed = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                batch_results = future.result()
                parsed.extend(batch_results)
                
                # Update progress more frequently and accurately
                processed_files += len(batches[batch_index])
                progress = int(10 + (processed_files / total_files * 60))  # Scale to 10-70% range
                emit_progress(progress)
                
                # Provide periodic updates for large jobs
                if batch_index % max(1, total_batches // 10) == 0 and batch_index > 0:
                    emit_log(f"Processed {processed_files}/{total_files} files ({progress}%)...")
        
        # FIXED: Improved grouping by directory, common base name, and extension
        groups = defaultdict(list)
        for d in parsed:
            key = (d['directory'], d['common_base'], d['ext'])
            groups[key].append(d)
        
        # Process each group of files
        result_rows = []
        emit_log(f"Processing {len(groups)} file groups")
        
        # Process file groups to detect sequences and single files
        progress_base = 70  # Start from 70% (after file processing)
        progress_step = 30 / max(1, len(groups))  # Remaining 30% divided by group count
        group_index = 0
        
        for key, items in groups.items():
            directory, base, ext = key
            
            # Update progress for each group
            group_index += 1
            emit_progress(int(progress_base + group_index * progress_step))
            
            # Separate sequences and single files
            seq_items = [i for i in items if i['frame'] is not None]
            single_items = [i for i in items if i['frame'] is None]
            
            # Process sequence items using optimized detection
            if seq_items and len(seq_items) >= self.settings.get("sequence_threshold", 5):
                # Use binary search for large sequences
                if self.binary_search and len(seq_items) > 20:
                    # Get a sample item for metadata
                    info = seq_items[0]
                    
                    # Find frame range using binary search
                    frame_nums = self.find_frame_range_binary_search(
                        directory,
                        base,
                        ext,
                        self.max_sampling_depth
                    )
                    
                    # Group continuous runs
                    runs = []
                    current_run = []
                    sequence_threshold = self.settings.get("sequence_threshold", 5)
                    
                    # Process the frame numbers into runs
                    for num in sorted(frame_nums):
                        if not current_run or num == current_run[-1] + 1:
                            current_run.append(num)
                        else:
                            if len(current_run) >= sequence_threshold:
                                runs.append(current_run)
                            current_run = [num]
                    
                    if current_run and len(current_run) >= sequence_threshold:
                        runs.append(current_run)
                    
                    # Create rows for each run
                    for run in runs:
                        if run:
                            # Find the actual file info for the first frame in the run
                            info = next((i for i in seq_items if i['frame'] == run[0]), seq_items[0])
                            
                            name = os.path.splitext(info['basename'])[0]
                            shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                            version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                            
                            # Create row for the sequence
                            row = [
                                name, shot, version, "", "",
                                ext.upper(), info['res'],
                                f"{len(run)} frames", datetime.now().strftime("%m/%d/%Y"),
                                delivery_package,
                                self.settings.get("upload_status"),
                                self.settings.get("vendor_name")
                            ]
                            emit_preview({"action": "update", "row_data": row})
                            result_rows.append(row)
                    
                    # Filter out frames that were part of runs
                    used_frames = set()
                    for run in runs:
                        used_frames.update(run)
                        
                    # Add remaining frames as individual items
                    for info in seq_items:
                        if info['frame'] is not None and info['frame'] not in used_frames:
                            name = os.path.splitext(info['basename'])[0]
                            shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                            version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                            
                            row = [
                                name, shot, version, "", "",
                                ext.upper(), info['res'],
                                "Still Frame", datetime.now().strftime("%m/%d/%Y"),
                                delivery_package,
                                self.settings.get("upload_status"),
                                self.settings.get("vendor_name")
                            ]
                            emit_preview({"action": "update", "row_data": row})
                            result_rows.append(row)
                else:
                    # Standard processing for smaller sequences
                    # Sort by frame number
                    seq_items.sort(key=lambda x: x['frame'])
                    
                    # Find runs of consecutive frames
                    runs = []
                    current_run = [seq_items[0]]
                    
                    for i in range(1, len(seq_items)):
                        if seq_items[i]['frame'] == current_run[-1]['frame'] + 1:
                            current_run.append(seq_items[i])
                        else:
                            if len(current_run) >= self.settings.get("sequence_threshold", 5):
                                runs.append(current_run)
                            current_run = [seq_items[i]]
                    
                    if len(current_run) >= self.settings.get("sequence_threshold", 5):
                        runs.append(current_run)
                    
                    # Create rows for runs that meet the threshold
                    for run in runs:
                        if run:
                            info = run[0]
                            name = os.path.splitext(info['basename'])[0]
                            shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                            version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                            
                            row = [
                                name, shot, version, "", "",
                                ext.upper(), info['res'],
                                f"{len(run)} frames", datetime.now().strftime("%m/%d/%Y"),
                                delivery_package,
                                self.settings.get("upload_status"),
                                self.settings.get("vendor_name")
                            ]
                            emit_preview({"action": "update", "row_data": row})
                            result_rows.append(row)
                    
                    # Process remaining frames as single items if they didn't form sequences
                    used_frames = set()
                    for run in runs:
                        used_frames.update(item['frame'] for item in run)
                    
                    remaining = [item for item in seq_items if item['frame'] not in used_frames]
                    
                    # Add the remaining items as single frames
                    for info in remaining:
                        name = os.path.splitext(info['basename'])[0]
                        shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                        version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                        
                        row = [
                            name, shot, version, "", "",
                            ext.upper(), info['res'],
                            "Still Frame", datetime.now().strftime("%m/%d/%Y"),
                            delivery_package,
                            self.settings.get("upload_status"),
                            self.settings.get("vendor_name")
                        ]
                        emit_preview({"action": "update", "row_data": row})
                        result_rows.append(row)
            else:
                # Add all seq_items as single frames if they don't meet threshold
                for info in seq_items:
                    name = os.path.splitext(info['basename'])[0]
                    shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                    version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                    
                    row = [
                        name, shot, version, "", "",
                        ext.upper(), info['res'],
                        "Still Frame", datetime.now().strftime("%m/%d/%Y"),
                        delivery_package,
                        self.settings.get("upload_status"),
                        self.settings.get("vendor_name")
                    ]
                    emit_preview({"action": "update", "row_data": row})
                    result_rows.append(row)
            
            # Process single items
            for info in single_items:
                name = os.path.splitext(info['basename'])[0]
                shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                version = 'v' + info['version'].zfill(self.settings.get("version_max"))
                
                # Get duration or file size
                if ext in self.video_types:
                    duration = self.get_file_duration(info['path'], ext)
                elif ext in self.image_types:
                    duration = "Still Frame"
                else:
                    duration = info['file_size']
                
                row = [
                    name, shot, version, "", "",
                    ext.upper(), info.get('res', 'N/A'),
                    duration, datetime.now().strftime("%m/%d/%Y"),
                    delivery_package,
                    self.settings.get("upload_status"),
                    self.settings.get("vendor_name")
                ]
                emit_preview({"action": "update", "row_data": row})
                result_rows.append(row)
        
        emit_progress(100)  # Complete!
        emit_preview({"action": "complete"})
        
        # Log completion statistics
        time_end = time.time()
        processing_time = time_end - time_start
        emit_log(f"Scan complete in {processing_time:.2f} seconds")
        emit_log(f"Found {len(result_rows)} valid items out of {total} candidate files")
        emit_log(f"Processing speed: {total/processing_time:.2f} files per second")
        
        return headers, result_rows

    def get_dir_listing(self, directory):
        """Cache directory listings to improve performance"""
        if directory not in self.dir_cache:
            try:
                self.dir_cache[directory] = os.listdir(directory)
            except Exception as e:
                self.dir_cache[directory] = []
        return self.dir_cache[directory]
    
    def validate_filename(self, name):
        """Validate filename against the pattern and return any errors"""
        # Fast validation cache check
        if name in self.fast_validation_cache:
            return self.fast_validation_cache[name]
            
        errors = []
        if '.' not in name:
            errors.append("Missing file extension")
            self.fast_validation_cache[name] = errors
            return errors
        
        m = self.pattern.match(name)
        if not m:
            errors.append("Filename does not match expected pattern")
            self.fast_validation_cache[name] = errors
            return errors
        
        g = m.groupdict()
        use_shot = self.settings.get("use_shot")
        
        if use_shot and g.get('shotNumber') and not re.match(r'^\d{4}$', g['shotNumber']):
            errors.append("Shot number must be exactly 4 digits if present")
        if not g['description']:
            errors.append("Description missing")
        if not re.match(r'^\d+[kK]$', g['resolution']):
            errors.append("Resolution must be digits+K")
        if not g['colorspaceGamma']:
            errors.append("ColorspaceGamma missing")
            
        ext = (g.get('extension') or g.get('extension2')).lower()
        fps = g.get('fps')
        
        # FIXED: Relaxed FPS validation for image sequences
        if self.settings.get("use_fps"):
            img_exts = self.image_types
            if ext in self.video_types and not fps:
                errors.append("FPS mandatory for video files")
        
        # Cache the result
        self.fast_validation_cache[name] = errors
        return errors
    
    def fast_extract_metadata(self, name):
        """
        Quickly extract metadata from a filename without full validation.
        This is much faster than the full process_file method for sequence detection.
        """
        m = self.pattern.match(name)
        if not m:
            return None
            
        g = m.groupdict()
        ext = (g.get('extension') or g.get('extension2')).lower()
        frame = int(g.get('frame_padding')) if g.get('frame_padding') else None
        common = re.sub(r'([_.]\d+)$','', os.path.splitext(name)[0]) if frame is not None else os.path.splitext(name)[0]
        
        return {
            'basename': name,
            'seq': g['sequence'],
            'shot': g.get('shotNumber') or '',
            'ext': ext,
            'res': g['resolution'],
            'frame': frame,
            'common_base': common,
            'version': g['version'],
            'fps': g.get('fps'),
        }
    
    def process_file(self, path):
        """Process a single file and return metadata if valid"""
        # Check if file is in cache to avoid redundant processing
        if path in self.file_cache:
            return self.file_cache[path]
            
        try:
            name = os.path.basename(path)
            
            # Fast check if the file exists before doing any expensive operations
            if not os.path.exists(path):
                result = {'valid': False, 'basename': name, 'errors': ["File does not exist"]}
                self.file_cache[path] = result
                return result
                
            # Special handling for other file types that don't need pattern validation
            ext = os.path.splitext(name)[1].lower()[1:] if '.' in name else ''
            if (self.settings.get("scan_other_types") and 
                ext in self.other_types and
                not (ext in self.image_types or ext in self.video_types)):
                result = self._process_other_file(path, name, ext)
                self.file_cache[path] = result
                return result
            
            # Standard pattern validation for media files
            errors = self.validate_filename(name)
            
            if errors:
                result = {'valid': False, 'basename': name, 'errors': errors}
                self.file_cache[path] = result
                return result
            
            metadata = self.fast_extract_metadata(name)
            if not metadata:
                result = {'valid': False, 'basename': name, 'errors': ["Failed to extract metadata"]}
                self.file_cache[path] = result
                return result
            
            # Add additional file information
            metadata.update({
                'path': path,
                'directory': os.path.dirname(path),
                'pixel_mapping': metadata.get('pixel_mapping', ''),
                'colorspace_gamma': metadata.get('colorspace_gamma', ''),
                'description': metadata.get('description', ''),
                'file_size': FileUtils.get_file_size(path),
                'file_type': 'Image' if ext in self.image_types else 'Video' if ext in self.video_types else 'Other'
            })
            
            result = {'valid': True, 'data': metadata}
            self.file_cache[path] = result
            return result
            
        except Exception as e:
            result = {'valid': False, 'basename': os.path.basename(path), 'errors': [f"Processing error: {str(e)}"]}
            self.file_cache[path] = result
            return result
    
    def _process_other_file(self, path, name, ext):
        """Process non-media files with different rules"""
        try:
            # Extract whatever we can from the filename 
            seq_match = re.search(r'^([A-Za-z]{1,5})(\d{4})?', name)
            if seq_match:
                seq = seq_match.group(1)
                shot = seq_match.group(2) if seq_match.group(2) else ""
            else:
                seq = "GEN"  # Generic sequence ID
                shot = ""
            
            # Look for something that might be a version number
            ver_match = re.search(r'v(\d{1,4})(?:_|\.)', name)
            version = ver_match.group(1).zfill(3) if ver_match else "001"  # Default version
            
            # Try to extract a description
            desc_match = re.search(r'_([a-zA-Z0-9-]+)_', name)
            description = desc_match.group(1) if desc_match else name.split('.')[0]
            
            return {'valid': True, 'data': {
                'path': path,
                'basename': name,
                'directory': os.path.dirname(path),
                'seq': seq,
                'shot': shot,
                'ext': ext,
                'res': 'N/A',
                'frame': None,
                'common_base': os.path.splitext(name)[0],
                'version': version,
                'fps': None,
                'pixel_mapping': '',
                'colorspace_gamma': 'N/A',
                'description': description,
                'file_size': FileUtils.get_file_size(path),
                'file_type': FileUtils.get_file_type(path)
            }}
        except Exception as e:
            return {'valid': False, 'basename': name, 'errors': [f"Processing error: {str(e)}"]}
    
    def find_frame_range_binary_search(self, directory, base_pattern, ext, max_depth=10):
        """
        Use binary search to quickly find the start and end frames of a sequence.
        Optimized version with better caching and faster sampling algorithm.
        """
        # Check if this sequence is already in cache
        cache_key = f"{directory}|{base_pattern}|{ext}"
        if cache_key in self.sequence_cache:
            return self.sequence_cache[cache_key]
            
        try:
            # If we can't access the directory, return empty range
            if not os.path.isdir(directory):
                return []
                
            # Generate the regex pattern for frame detection
            pat = re.compile(r'^' + re.escape(base_pattern) + r'[_.](\d+)\.' + re.escape(ext) + r'$', re.IGNORECASE)
            
            # Get all filenames in directory (from cache if possible)
            all_files = self.get_dir_listing(directory)
            
            # First pass: quickly identify frame numbers using regex
            frame_files = {}
            
            # Optimize with set operations for large directories
            if len(all_files) > 1000:
                # First do a quick string contains check to filter out obvious non-matches
                base_simple = os.path.basename(base_pattern)
                potential_matches = [f for f in all_files if base_simple in f and f.endswith(f'.{ext}')]
                
                # Then do the more expensive regex match
                for filename in potential_matches:
                    m = pat.match(filename)
                    if m:
                        try:
                            frame_num = int(m.group(1))
                            frame_files[frame_num] = filename
                        except (ValueError, IndexError):
                            pass
            else:
                # For smaller directories, just do the regex directly
                for filename in all_files:
                    m = pat.match(filename)
                    if m:
                        try:
                            frame_num = int(m.group(1))
                            frame_files[frame_num] = filename
                        except (ValueError, IndexError):
                            pass
            
            # If we have less than a handful of files, just use them directly
            if len(frame_files) <= 20:
                frame_numbers = sorted(frame_files.keys())
                self.sequence_cache[cache_key] = frame_numbers
                return frame_numbers
                
            # Get sorted frame numbers
            frame_numbers = sorted(frame_files.keys())
            if not frame_numbers:
                return []
                
            # Get min and max frames
            min_frame = min(frame_numbers)
            max_frame = max(frame_numbers)
            
            # If range is tightly packed (first and last differ by about the count),
            # it's likely a continuous sequence without many gaps
            if max_frame - min_frame < len(frame_numbers) * 1.2:
                # Just check if we have every frame in the range
                complete_sequence = True
                for f in range(min_frame, max_frame + 1):
                    if f not in frame_files:
                        complete_sequence = False
                        break
                        
                if complete_sequence:
                    # It's a complete sequence, return the range
                    frame_numbers = list(range(min_frame, max_frame + 1))
                    self.sequence_cache[cache_key] = frame_numbers
                    return frame_numbers
            
            # For large ranges, use optimized binary search strategy
            if max_frame - min_frame > 1000 and self.binary_search:
                # Start with known min and max
                confirmed_frames = set([min_frame, max_frame])
                
                # Add some samples between min and max for better initial coverage
                # Use log-spaced samples for better distribution
                range_size = max_frame - min_frame
                if range_size > 10:
                    # Add logarithmically spaced samples
                    import math
                    num_samples = min(20, max(5, int(math.log2(range_size))))
                    step = range_size / (num_samples + 1)
                    
                    for i in range(1, num_samples + 1):
                        sample = min_frame + int(i * step)
                        if sample in frame_files:
                            confirmed_frames.add(sample)
                
                # Binary search for gaps
                gaps_to_check = [(min_frame, max_frame)]
                checked_gaps = set()
                
                # Limit iterations to avoid infinite loops
                max_iterations = min(1000, range_size // 2)
                iterations = 0
                
                while gaps_to_check and iterations < max_iterations:
                    start, end = gaps_to_check.pop(0)
                    iterations += 1
                    
                    # Skip if we've checked this gap before
                    if (start, end) in checked_gaps or end - start <= 1:
                        continue
                        
                    checked_gaps.add((start, end))
                    
                    # Find middle point
                    mid = start + (end - start) // 2
                    
                    # Check if mid frame exists
                    if mid in frame_files:
                        confirmed_frames.add(mid)
                        
                        # Add the two halves as gaps to check
                        if mid - start > 1:
                            gaps_to_check.append((start, mid))
                        if end - mid > 1:
                            gaps_to_check.append((mid, end))
                    else:
                        # Frame doesn't exist, could be a gap
                        # Check surrounding frames to determine gap boundaries
                        left_found = False
                        right_found = False
                        
                        # Look left for nearest existing frame
                        for left in range(mid-1, start, -1):
                            if left in frame_files:
                                confirmed_frames.add(left)
                                left_found = True
                                
                                # Check right half of this new gap
                                if mid - left > 1:
                                    gaps_to_check.append((left, mid))
                                break
                                
                        # Look right for nearest existing frame
                        for right in range(mid+1, end):
                            if right in frame_files:
                                confirmed_frames.add(right)
                                right_found = True
                                
                                # Check left half of this new gap
                                if right - mid > 1:
                                    gaps_to_check.append((mid, right))
                                break
                        
                        # If we didn't find frames on either side, this is a large gap
                        if not left_found and not right_found:
                            # Skip this gap, it's likely part of a larger gap
                            pass
                
                # Convert confirmed frames to a sorted list
                frame_numbers = sorted(confirmed_frames)
                
                # Verify continuity to find runs
                runs = []
                current_run = []
                
                for frame in frame_numbers:
                    if not current_run or frame == current_run[-1] + 1:
                        current_run.append(frame)
                    else:
                        # Verify frames in this run actually exist
                        verified = [f for f in current_run if f in frame_files]
                        if verified:
                            runs.append(verified)
                        current_run = [frame]
                        
                if current_run:
                    verified = [f for f in current_run if f in frame_files]
                    if verified:
                        runs.append(verified)
                
                # Flatten runs into frame numbers
                frame_numbers = []
                for run in runs:
                    if len(run) >= self.settings.get("sequence_threshold", 5):
                        # For runs that meet the threshold, fill in any small gaps
                        start, end = run[0], run[-1]
                        continuous_run = []
                        
                        # If the gap is small, add all frames in between for continuity
                        if end - start < 20:
                            continuous_run = list(range(start, end + 1))
                        else:
                            continuous_run = run
                            
                        frame_numbers.extend(continuous_run)
                    else:
                        # For small runs, only include verified frames
                        frame_numbers.extend(run)
                
                # Sort the final list
                frame_numbers.sort()
                
            # Cache the result
            self.sequence_cache[cache_key] = frame_numbers
            return frame_numbers
            
        except Exception as e:
            logger.warning(f"Error in binary search for sequence {base_pattern}.{ext}: {e}")
            return list(sorted(frame_files.keys())) if frame_files else []
        
        def _sample_range(self, min_val, max_val, max_depth):
            """
            Use binary sampling to efficiently check a range.
            Returns a list of sample points to check.
            """
            samples = [min_val, max_val]
            
            def sample_recursive(start, end, depth):
                if depth <= 0 or end - start <= 1:
                    return
                    
                mid = start + (end - start) // 2
                samples.append(mid)
                sample_recursive(start, mid, depth - 1)
                sample_recursive(mid, end, depth - 1)
                
            sample_recursive(min_val, max_val, max_depth)
            return sorted(samples)
        
        def _find_consecutive_runs(self, sorted_values):
            """
            Find consecutive runs in a sorted list of values.
            Returns a list of lists, where each inner list is a consecutive run.
            """
            if not sorted_values:
                return []
                
            runs = []
            current_run = [sorted_values[0]]
            
            for i in range(1, len(sorted_values)):
                # If consecutive
                if sorted_values[i] == current_run[-1] + 1:
                    current_run.append(sorted_values[i])
                else:
                    runs.append(current_run)
                    current_run = [sorted_values[i]]
                    
            if current_run:
                runs.append(current_run)
                
            return runs
            
def get_file_duration(self, file_path, ext):
    """
    Get duration for video files or return 'Still Frame' for images.
    Enhanced to properly handle large video files while still reporting frame count.
    """
    try:
        if ext.lower() not in self.image_types:
            # Import cv2 only if we need it (for CLI mode)
            import cv2
            
            # Check if this is a large file
            is_large_file = FileUtils.is_large_file(file_path, self.file_size_warning)
            
            # For large files in safe mode, use a more careful approach
            if self.safe_mode and is_large_file:
                try:
                    # Use thread-safe timeout mechanism with a shorter timeout for safety
                    with ThreadSafeTimeout(15):  # 15 second timeout for large files
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            return "Error Opening Large File"
                            
                        # Only get essential properties to avoid lengthy operations
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Release immediately to free resources
                        cap.release()
                        
                        # Even for large files, return the frame count
                        return f"Large File: {frame_count} frames"
                except TimeoutError:
                    return "Timeout: Large Video File"
                except Exception as e:
                    return f"Large Video File (~{FileUtils.get_file_size(file_path)})"
            else:
                # Standard handling for normal-sized files or when safe mode is off
                with ThreadSafeTimeout(30):  # 30 second timeout
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        return "Unknown"
                        
                    # Get fps and frame count for more accurate duration
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if fps > 0 and frame_count > 0:
                        # Calculate duration in seconds
                        duration_sec = frame_count / fps
                        # Format as MM:SS if more than a minute
                        if duration_sec >= 60:
                            minutes = int(duration_sec // 60)
                            seconds = int(duration_sec % 60)
                            duration_str = f"{minutes}:{seconds:02d} ({frame_count} frames)"
                        else:
                            duration_str = f"{int(duration_sec)}s ({frame_count} frames)"
                    else:
                        duration_str = f"{frame_count} frames"
                    
                    cap.release()
                    return duration_str
                    
        else:
            return "Still Frame"
    except TimeoutError:
        return "Duration Check Timeout"
    except Exception as e:
        logger.warning(f"Error getting duration for {file_path}: {str(e)}")
        return "Error"            
def save_csv(filename, headers, rows):
    """Save data to CSV file"""
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        logger.info(f"CSV saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        return False

# --------------------
# GUI Classes (only imported if GUI_AVAILABLE)
# --------------------
if GUI_AVAILABLE:
    class FolderDropListWidget(QListWidget):
        """List widget that supports drag & drop of folders"""
        def __init__(self, parent=None):
            super(FolderDropListWidget, self).__init__(parent)
            self.setSelectionMode(QListWidget.ExtendedSelection)
            self.setAcceptDrops(True)
            self.setDragDropMode(QListWidget.DropOnly)

        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()
            else:
                e.ignore()

        def dragMoveEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()
            else:
                e.ignore()

        def dropEvent(self, e):
            if e.mimeData().hasUrls():
                for url in e.mimeData().urls():
                    p = url.toLocalFile()
                    if os.path.isdir(p) and p not in [self.item(i).text() for i in range(self.count())]:
                        it = QListWidgetItem(p)
                        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                        it.setCheckState(Qt.Checked)
                        self.addItem(it)
                    elif os.path.isfile(p):
                        # Allow dropping files directly into folder list
                        folder_path = os.path.dirname(p)
                        if folder_path and folder_path not in [self.item(i).text() for i in range(self.count())]:
                            it = QListWidgetItem(folder_path)
                            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                            it.setCheckState(Qt.Checked)
                            self.addItem(it)
                e.acceptProposedAction()
            else:
                e.ignore()

    class BatchEditDialog(QDialog):
        """Dialog for batch editing selected cells"""
        def __init__(self, column_name, current_values, custom_values, parent=None):
            super(BatchEditDialog, self).__init__(parent)
            self.setWindowTitle(f"Edit {column_name}")
            self.resize(400, 300)
            
            layout = QVBoxLayout()
            
            # Show current values
            if current_values:
                current_group = QGroupBox("Current Values")
                current_layout = QVBoxLayout()
                current_text = QLabel(", ".join(current_values))
                current_text.setWordWrap(True)
                current_layout.addWidget(current_text)
                current_group.setLayout(current_layout)
                layout.addWidget(current_group)
            
            # Input for new value
            input_group = QGroupBox("Enter New Value")
            input_layout = QVBoxLayout()
            
            self.combobox = QComboBox()
            self.combobox.setEditable(True)
            
            # Add custom values to the dropdown
            if custom_values:
                self.combobox.addItems(custom_values)
            
            # Add current values if they're not already in custom_values
            for val in current_values:
                if val and val not in custom_values:
                    self.combobox.addItem(val)
            
            input_layout.addWidget(self.combobox)
            input_group.setLayout(input_layout)
            layout.addWidget(input_group)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("Apply")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(self.accept)
            cancel_button.clicked.connect(self.reject)
            
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)
            
            self.setLayout(layout)
        
        def get_value(self):
            return self.combobox.currentText()

    class ResultsTableWidget(QTableWidget):
        """Enhanced table widget for scan results with editing capabilities"""
        def __init__(self, parent=None):
            super(ResultsTableWidget, self).__init__(parent)
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.show_context_menu)
            self.settings = None  # Will be set by parent
            
            # Configure the table
            self.setSortingEnabled(True)
            self.setSelectionBehavior(QTableWidget.SelectRows)
            self.setSelectionMode(QTableWidget.ExtendedSelection)
            
            # Make specific columns resizable
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.horizontalHeader().setStretchLastSection(True)
            
            # Set minimum width for all columns
            self.horizontalHeader().setMinimumSectionSize(120)
            
            # Make the table take up available space
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            # Set fixed size for some columns
            self.column_widths = {
                "Version Name": 200,  # Column 0
                "Shot Name": 150,     # Column 1
                "Version Number": 120, # Column 2
                "File Type": 100,      # Column 5
                "Resolution": 120,     # Column 6
                "Duration/Size": 150,  # Column 7
                "Delivery Date": 120   # Column 8
            }
        
        def setup_columns(self):
            """Set up column widths after headers are set"""
            # Adjust horizontal header sizes
            self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            
            # Configure each column
            for col, header in enumerate(self.column_headers):
                if header in self.column_widths:
                    self.setColumnWidth(col, self.column_widths[header])
        
        def update_settings(self, settings):
            """Update settings reference"""
            self.settings = settings
        
        def show_context_menu(self, pos):
            """Show context menu with editing options"""
            if not self.settings:
                return
                
            selected_indexes = self.selectedIndexes()
            if not selected_indexes:
                return
                
            # Group selected indexes by column
            columns = {}
            for idx in selected_indexes:
                col = idx.column()
                if col not in columns:
                    columns[col] = []
                columns[col].append(idx)
            
            menu = QMenu(self)
            
            # Only add batch edit option for specific columns
            editable_columns = {
                3: "Submitted For",  # Column index for "Submitted For"
                4: "Delivery Notes",  # Column index for "Delivery Notes"
                9: "Delivery Package Name",  # Column index for "Delivery Package Name"
                10: "Upload Status",  # Column index for "Upload Status"
                11: "Vendor Name"  # Column index for "Vendor Name"
            }
            
            # Add batch edit options for each editable column that has selected cells
            for col, name in editable_columns.items():
                if col in columns:
                    action = QAction(f"Edit {name} for Selected Rows", self)
                    action.triggered.connect(lambda checked=False, col=col, name=name: 
                                             self.batch_edit_column(col, name))
                    menu.addAction(action)
            
            # Add copy options
            menu.addSeparator()
            copy_action = QAction("Copy Selected Cell(s)", self)
            copy_action.triggered.connect(self.copy_selected_cells)
            menu.addAction(copy_action)
            
            copy_row_action = QAction("Copy Selected Row(s)", self)
            copy_row_action.triggered.connect(self.copy_selected_rows)
            menu.addAction(copy_row_action)
            
            # Add delete option
            menu.addSeparator()
            delete_action = QAction("Delete Selected Row(s)", self)
            delete_action.triggered.connect(self.delete_selected_rows)
            menu.addAction(delete_action)
            
            # Show menu at cursor position
            menu.exec_(self.viewport().mapToGlobal(pos))
        
        def batch_edit_column(self, col, name):
            """Batch edit all selected cells in a column"""
            if not self.settings:
                return
                
            # Get all selected rows for this column
            selected_rows = set()
            current_values = set()
            
            for idx in self.selectedIndexes():
                if idx.column() == col:
                    selected_rows.add(idx.row())
                    item = self.item(idx.row(), col)
                    if item:
                        current_values.add(item.text())
            
            if not selected_rows:
                return
            
            # Get custom values for this column
            custom_values = []
            if name == "Submitted For":
                custom_values = self.settings.get("custom_submitted_for", [])
            elif name == "Delivery Notes":
                custom_values = self.settings.get("custom_delivery_notes", [])
            
            # Show dialog for editing
            dialog = BatchEditDialog(name, list(current_values), custom_values, self)
            if dialog.exec_() == QDialog.Accepted:
                new_value = dialog.get_value()
                
                # Update all selected cells
                for row in selected_rows:
                    self.setItem(row, col, QTableWidgetItem(new_value))
                
                # Save custom value for future use if it's not already in the list
                if name == "Submitted For" and new_value and new_value not in custom_values:
                    custom_values = self.settings.get("custom_submitted_for", [])
                    if new_value not in custom_values:
                        custom_values.append(new_value)
                        self.settings.set("custom_submitted_for", custom_values)
                        self.settings.save()
                elif name == "Delivery Notes" and new_value and new_value not in custom_values:
                    custom_values = self.settings.get("custom_delivery_notes", [])
                    if new_value not in custom_values:
                        custom_values.append(new_value)
                        self.settings.set("custom_delivery_notes", custom_values)
                        self.settings.save()
        
        def copy_selected_cells(self):
            """Copy selected cells to clipboard"""
            selected = self.selectedIndexes()
            if not selected:
                return
                
            text = ""
            prev_row = selected[0].row()
            for idx in selected:
                if idx.row() != prev_row:
                    text += "\n"
                    prev_row = idx.row()
                elif text:
                    text += "\t"
                    
                item = self.item(idx.row(), idx.column())
                text += item.text() if item else ""
                
            QApplication.clipboard().setText(text)
        
        def copy_selected_rows(self):
            """Copy entire selected rows to clipboard"""
            rows = set()
            for idx in self.selectedIndexes():
                rows.add(idx.row())
                
            if not rows:
                return
                
            text = ""
            for row in sorted(rows):
                if text:
                    text += "\n"
                row_text = []
                for col in range(self.columnCount()):
                    item = self.item(row, col)
                    row_text.append(item.text() if item else "")
                text += "\t".join(row_text)
                
            QApplication.clipboard().setText(text)
        
        def delete_selected_rows(self):
            """Delete selected rows from the table"""
            rows = sorted(set(idx.row() for idx in self.selectedIndexes()), reverse=True)
            if not rows:
                return
                
            for row in rows:
                self.removeRow(row)

    class FileScanner(QThread):
        """Thread for scanning files without blocking the UI"""
        progress = pyqtSignal(int)
        update_preview = pyqtSignal(dict)
        log_message = pyqtSignal(str)
        scan_complete = pyqtSignal(list, list)  # headers, rows for saving

        def __init__(self, processor, directories, delivery_package, file_types, include_other_types=False):
            super(FileScanner, self).__init__()
            self.processor = processor
            self.directories = directories
            self.delivery_package = delivery_package
            self.file_types = file_types
            self.include_other_types = include_other_types
            self.headers = []
            self.rows = []

        def run(self):
            try:
                self.headers, self.rows = self.processor.scan_directories(
                    self.directories,
                    self.file_types,
                    self.delivery_package,
                    self.progress.emit,
                    self.log_message.emit,
                    self.update_preview.emit,
                    self.include_other_types
                )
                self.scan_complete.emit(self.headers, self.rows)
            except Exception as e:
                self.log_message.emit(f"Error in scanner thread: {e}")
                self.log_message.emit(traceback.format_exc())
                self.scan_complete.emit([], [])  # Empty results on error
    class MainWindow(QMainWindow):
        """Main application window"""
        def __init__(self):
            try:
                super(MainWindow, self).__init__()
                self.setWindowTitle(f"File Scanner v{VERSION} - CSV Creator")
                
                # Initialize early for error cases
                self.log_window = None
                
                self.settings = Settings()
                self.processor = FileProcessor(self.settings)
                self.scanner = None
                self.imageTypeCheckboxes = {}
                self.videoTypeCheckboxes = {}
                self.otherTypeCheckboxes = {}
                
                self.setGeometry(100, 100, 1200, 900)
                
                # Restore window geometry if available
                geometry = self.settings.get("window_geometry")
                if geometry:
                    try:
                        self.restoreGeometry(bytes.fromhex(geometry))
                    except Exception as e:
                        logger.error(f"Error restoring window geometry: {e}")
                
                # Create menus
                self.create_menus()
                
                # Status bar
                self.status_bar = self.statusBar()
                self.status_bar.showMessage("Ready - drag and drop folders below.")
                
                # Main tab widget
                self.tab_widget = QTabWidget()
                self.create_scan_tab()
                self.create_results_tab()
                self.create_settings_tab()
                self.create_advanced_tab()
                self.create_help_tab()
                
                # Set as central widget
                self.setCentralWidget(self.tab_widget)
                self.setStyleSheet(DARK_STYLESHEET)
                
                # Populate from settings
                self.load_from_settings()
                
                # Log startup success
                logger.info("GUI initialized successfully")
                if self.log_window:
                    self.log_window.appendPlainText(f"Application started successfully - v{VERSION}")
                    self.log_window.appendPlainText("High-performance binary search enabled for large sequences")
                
            except Exception as e:
                # Handle initialization errors
                logger.error(f"Error initializing MainWindow: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("Application Initialization Error", 
                                      f"Error initializing the application: {e}\n\n"
                                      "Please check log files in your home directory.")

        def show_error_dialog(self, title, message):
            """Show error dialog with detailed message"""
            try:
                error_dialog = QErrorMessage(self)
                error_dialog.setWindowTitle(title)
                error_dialog.showMessage(message)
            except Exception:
                # Fallback if QErrorMessage fails
                QMessageBox.critical(None, title, message)

        def create_menus(self):
            """Create application menu bar"""
            try:
                menubar = self.menuBar()
                
                # File menu
                file_menu = menubar.addMenu('&File')
                
                add_dir_action = QAction('&Add Directory...', self)
                add_dir_action.triggered.connect(self.add_directory)
                file_menu.addAction(add_dir_action)
                
                file_menu.addSeparator()
                
                save_csv_action = QAction('&Save CSV...', self)
                save_csv_action.triggered.connect(self.save_csv)
                file_menu.addAction(save_csv_action)
                
                file_menu.addSeparator()
                
                exit_action = QAction('E&xit', self)
                exit_action.triggered.connect(self.close)
                file_menu.addAction(exit_action)
                
                # Edit menu
                edit_menu = menubar.addMenu('&Edit')
                
                clear_folders_action = QAction('&Clear All Folders', self)
                clear_folders_action.triggered.connect(self.clear_folders)
                edit_menu.addAction(clear_folders_action)
                
                # Tools menu
                tools_menu = menubar.addMenu('&Tools')
                
                refresh_action = QAction('&Refresh File List', self)
                refresh_action.triggered.connect(self.refresh_file_list)
                tools_menu.addAction(refresh_action)
                
                recursive_scan_action = QAction('&Toggle Recursive Scan', self, checkable=True)
                recursive_scan_action.setChecked(self.settings.get("recursive_scan", True))
                recursive_scan_action.triggered.connect(self.toggle_recursive_scan)
                tools_menu.addAction(recursive_scan_action)
                
                safe_mode_action = QAction('&Toggle Safe Mode', self, checkable=True)
                safe_mode_action.setChecked(self.settings.get("safe_mode", True))
                safe_mode_action.triggered.connect(self.toggle_safe_mode)
                tools_menu.addAction(safe_mode_action)
                
                binary_search_action = QAction('&Toggle Binary Search', self, checkable=True)
                binary_search_action.setChecked(self.settings.get("binary_search", True))
                binary_search_action.triggered.connect(self.toggle_binary_search)
                tools_menu.addAction(binary_search_action)
                
                # Help menu
                help_menu = menubar.addMenu('&Help')
                
                about_action = QAction('&About', self)
                about_action.triggered.connect(self.show_about)
                help_menu.addAction(about_action)
                
                debug_action = QAction('Show &Debug Info', self)
                debug_action.triggered.connect(self.show_debug_info)
                help_menu.addAction(debug_action)
                
            except Exception as e:
                logger.error(f"Error creating menus: {e}")
                logger.error(traceback.format_exc())
        
        def toggle_recursive_scan(self, state):
            """Toggle recursive scanning of subdirectories"""
            self.settings.set("recursive_scan", state)
            self.settings.save()
            self.status_bar.showMessage(
                f"Recursive scanning {'enabled' if state else 'disabled'}", 3000
            )
            
        def toggle_safe_mode(self, state):
            """Toggle safe mode for large file handling"""
            self.settings.set("safe_mode", state)
            self.processor.safe_mode = state
            self.settings.save()
            self.status_bar.showMessage(
                f"Safe mode {'enabled' if state else 'disabled'} for large file handling", 3000
            )
            if not state:
                QMessageBox.warning(self, "Safe Mode Disabled", 
                                  "Disabling safe mode may cause crashes when processing very large files.\n"
                                  "Only disable safe mode if you're experiencing issues with normal files.")
        
        def toggle_binary_search(self, state):
            """Toggle binary search for sequences"""
            self.settings.set("binary_search", state)
            self.processor.binary_search = state
            self.settings.save()
            self.status_bar.showMessage(
                f"Binary search {'enabled' if state else 'disabled'} for sequences", 3000
            )
            if not state:
                QMessageBox.warning(self, "Binary Search Disabled", 
                                  "Disabling binary search will make scanning much slower for large sequences.\n"
                                  "Only disable binary search if you're experiencing issues with sequence detection.")

        def create_scan_tab(self):
            """Create the main scanning tab"""
            try:
                scan_tab = QWidget()
                layout = QVBoxLayout()
                
                # Folder section
                folder_group = QGroupBox("Drag & Drop Folders")
                self.folder_list = FolderDropListWidget()
                self.folder_list.setMinimumHeight(100)
                folder_layout = QVBoxLayout()
                folder_layout.addWidget(self.folder_list)
                folder_group.setLayout(folder_layout)
                
                folder_btn_layout = QHBoxLayout()
                add_btn = QPushButton("Add Folder...")
                add_btn.clicked.connect(self.add_directory)
                remove_btn = QPushButton("Remove Selected")
                remove_btn.clicked.connect(self.remove_selected_folders)
                clear_btn = QPushButton("Clear All")
                clear_btn.clicked.connect(self.clear_folders)
                folder_btn_layout.addWidget(add_btn)
                folder_btn_layout.addWidget(remove_btn)
                folder_btn_layout.addWidget(clear_btn)
                
                # File-type selectors
                type_group = QGroupBox("File Type Selection")
                type_layout = QVBoxLayout()
                
                selector_layout = QHBoxLayout()
                self.select_all = QCheckBox("Select All File Types")
                self.select_img = QCheckBox("Select All Image Files")
                self.select_vid = QCheckBox("Select All Video Files")
                self.select_other = QCheckBox("Select Other File Types")
                
                self.select_all.toggled.connect(self.on_select_all)
                self.select_img.toggled.connect(self.on_select_image)
                self.select_vid.toggled.connect(self.on_select_video)
                self.select_other.toggled.connect(self.on_select_other)
                
                selector_layout.addWidget(self.select_all)
                selector_layout.addWidget(self.select_img)
                selector_layout.addWidget(self.select_vid)
                selector_layout.addWidget(self.select_other)
                type_layout.addLayout(selector_layout)
                
                type_subgroups = QHBoxLayout()
                
                # Image types
                img_box = QGroupBox("Image File Types")
                img_layout = QHBoxLayout()
                for ext in self.settings.get("image_types"):
                    cb = QCheckBox(ext.upper())
                    cb.setChecked(True)
                    img_layout.addWidget(cb)
                    self.imageTypeCheckboxes[ext] = cb
                img_box.setLayout(img_layout)
                type_subgroups.addWidget(img_box)
                
                # Video types
                vid_box = QGroupBox("Video File Types")
                vid_layout = QHBoxLayout()
                for ext in self.settings.get("video_types"):
                    cb = QCheckBox(ext.upper())
                    cb.setChecked(True)
                    vid_layout.addWidget(cb)
                    self.videoTypeCheckboxes[ext] = cb
                vid_box.setLayout(vid_layout)
                type_subgroups.addWidget(vid_box)
                
                # Other types
                other_box = QGroupBox("Other File Types")
                other_layout = QHBoxLayout()
                
                # Create a grid layout with 3 columns for other types
                other_grid = QGridLayout()
                row, col = 0, 0
                
                for ext in self.settings.get("other_types"):
                    cb = QCheckBox(ext.upper())
                    cb.setChecked(self.settings.get("scan_other_types", False))
                    other_grid.addWidget(cb, row, col)
                    self.otherTypeCheckboxes[ext] = cb
                    
                    # Move to next column or row
                    col += 1
                    if col > 2:  # 3 columns (0, 1, 2)
                        col = 0
                        row += 1
                
                other_layout.addLayout(other_grid)
                other_box.setLayout(other_layout)
                
                # Add to main layout with a splitter for resizing
                type_splitter = QSplitter(Qt.Horizontal)
                
                # Create widgets to hold each group
                img_container = QWidget()
                img_container.setLayout(QVBoxLayout())
                img_container.layout().addWidget(img_box)
                
                vid_container = QWidget()
                vid_container.setLayout(QVBoxLayout())
                vid_container.layout().addWidget(vid_box)
                
                other_container = QWidget()
                other_container.setLayout(QVBoxLayout())
                other_container.layout().addWidget(other_box)
                
                type_splitter.addWidget(img_container)
                type_splitter.addWidget(vid_container)
                type_splitter.addWidget(other_container)
                
                type_layout.addWidget(type_splitter)
                type_group.setLayout(type_layout)
                
                # Delivery package
                del_group = QGroupBox("Delivery Package")
                del_layout = QHBoxLayout()
                del_layout.addWidget(QLabel("Delivery Package Name:"))
                self.delEdit = QLineEdit()
                self.delEdit.textChanged.connect(self.update_delivery_field)
                del_layout.addWidget(self.delEdit)
                del_group.setLayout(del_layout)
                
                # Progress bar
                progress_group = QGroupBox("Scan Progress")
                progress_layout = QVBoxLayout()
                self.progress_bar = QProgressBar()
                self.progress_bar.setMaximum(100)
                progress_layout.addWidget(self.progress_bar)
                progress_group.setLayout(progress_layout)
                
                # Action buttons
                btn_layout = QHBoxLayout()
                self.start_btn = QPushButton("Start Scan")
                self.start_btn.clicked.connect(self.start_scan)
                self.save_btn = QPushButton("Save CSV")
                self.save_btn.clicked.connect(self.save_csv)
                self.save_btn.setEnabled(False) # Disable until scan completes
                btn_layout.addWidget(self.start_btn)
                btn_layout.addWidget(self.save_btn)
                
                # Log window
                log_group = QGroupBox("Log")
                log_layout = QVBoxLayout()
                self.log_window = QPlainTextEdit()
                self.log_window.setReadOnly(True)
                self.log_window.setMaximumHeight(150)
                log_layout.addWidget(self.log_window)
                log_group.setLayout(log_layout)
                
                # Assemble the layout
                layout.addWidget(folder_group)
                layout.addLayout(folder_btn_layout)
                layout.addWidget(type_group)
                layout.addWidget(del_group)
                layout.addWidget(progress_group)
                layout.addLayout(btn_layout)
                layout.addWidget(log_group)
                
                scan_tab.setLayout(layout)
                self.tab_widget.addTab(scan_tab, "Scan Files")
            
            except Exception as e:
                logger.error(f"Error creating scan tab: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("GUI Error", f"Error creating scan tab: {e}")
                
        def create_results_tab(self):
            """Create a dedicated tab for scan results"""
            try:
                results_tab = QWidget()
                layout = QVBoxLayout()
                
                # Create enhanced table widget for results
                self.results_table = ResultsTableWidget()
                self.results_table.update_settings(self.settings)
                
                # Set column headers - we'll use the same headers as the preview table
                self.column_headers = [
                    "Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes",
                    "File Type", "Resolution", "Duration/Size", "Delivery Date",
                    "Delivery Package Name", "Upload Status", "Vendor Name"
                ]
                self.results_table.setColumnCount(len(self.column_headers))
                self.results_table.setHorizontalHeaderLabels(self.column_headers)
                self.results_table.column_headers = self.column_headers
                self.results_table.setup_columns()
                
                # Info label with instructions
                info_label = QLabel(
                    "Right-click on selected rows to edit values or use the buttons below. "
                    "Multi-select is supported for batch editing."
                )
                info_label.setWordWrap(True)
                
                # Quick edit buttons for common fields
                edit_button_layout = QHBoxLayout()
                
                submitted_for_btn = QPushButton("Edit Submitted For")
                submitted_for_btn.clicked.connect(lambda: self.edit_column_for_selected(3, "Submitted For"))
                
                delivery_notes_btn = QPushButton("Edit Delivery Notes")
                delivery_notes_btn.clicked.connect(lambda: self.edit_column_for_selected(4, "Delivery Notes"))
                
                edit_button_layout.addWidget(submitted_for_btn)
                edit_button_layout.addWidget(delivery_notes_btn)
                
                # Save button
                save_layout = QHBoxLayout()
                save_results_btn = QPushButton("Save CSV from Results")
                save_results_btn.clicked.connect(self.save_csv_from_results)
                save_layout.addWidget(save_results_btn)
                
                # Assemble the layout
                layout.addWidget(info_label)
                layout.addWidget(self.results_table)
                layout.addLayout(edit_button_layout)
                layout.addLayout(save_layout)
                
                results_tab.setLayout(layout)
                self.tab_widget.addTab(results_tab, "Scan Results")
                
                # Initially disable the tab until we have results
                self.tab_widget.setTabEnabled(self.tab_widget.indexOf(results_tab), False)
                
            except Exception as e:
                logger.error(f"Error creating results tab: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("GUI Error", f"Error creating results tab: {e}")
                
        def edit_column_for_selected(self, col, name):
            """Edit a specific column for all selected rows"""
            self.results_table.batch_edit_column(col, name)

        def create_settings_tab(self):
            """Create the settings tab"""
            try:
                settings_tab = QWidget()
                layout = QVBoxLayout()
                
                # Pattern settings
                pat_group = QGroupBox("Filename Pattern Settings")
                pat_layout = QGridLayout()
                
                pat_layout.addWidget(QLabel("Sequence Length (min – max):"), 0, 0)
                self.seqMin = QSpinBox()
                self.seqMin.setRange(1, 10)
                self.seqMin.setValue(self.settings.get("sequence_min"))
                pat_layout.addWidget(self.seqMin, 0, 1)
                
                self.seqMax = QSpinBox()
                self.seqMax.setRange(1, 10)
                self.seqMax.setValue(self.settings.get("sequence_max"))
                pat_layout.addWidget(self.seqMax, 0, 2)
                
                self.shotCheck = QCheckBox("Enable 4-digit shot number")
                self.shotCheck.setChecked(self.settings.get("use_shot"))
                pat_layout.addWidget(self.shotCheck, 1, 0, 1, 3)
                
                self.pixCheck = QCheckBox("Enable LL180/360 pixel mapping")
                self.pixCheck.setChecked(self.settings.get("use_pixel"))
                pat_layout.addWidget(self.pixCheck, 2, 0, 1, 3)
                
                self.fpsCheck = QCheckBox("Enable FPS component")
                self.fpsCheck.setChecked(self.settings.get("use_fps"))
                pat_layout.addWidget(self.fpsCheck, 3, 0, 1, 3)
                
                pat_layout.addWidget(QLabel("Version Digits (min – max):"), 4, 0)
                self.verMin = QSpinBox()
                self.verMin.setRange(1, 10)
                self.verMin.setValue(self.settings.get("version_min"))
                pat_layout.addWidget(self.verMin, 4, 1)
                
                self.verMax = QSpinBox()
                self.verMax.setRange(1, 10)
                self.verMax.setValue(self.settings.get("version_max"))
                pat_layout.addWidget(self.verMax, 4, 2)
                
                pat_layout.addWidget(QLabel("Minimum frames for sequence:"), 5, 0)
                self.seqThreshold = QSpinBox()
                self.seqThreshold.setRange(2, 20)
                self.seqThreshold.setValue(self.settings.get("sequence_threshold"))
                pat_layout.addWidget(self.seqThreshold, 5, 1, 1, 2)
                
                pat_group.setLayout(pat_layout)
                
                # Output settings
                out_group = QGroupBox("Output Settings")
                out_layout = QGridLayout()
                
                out_layout.addWidget(QLabel("Vendor Name:"), 0, 0)
                self.vendorEdit = QLineEdit(self.settings.get("vendor_name"))
                out_layout.addWidget(self.vendorEdit, 0, 1)
                
                out_layout.addWidget(QLabel("Upload Status:"), 1, 0)
                self.uploadEdit = QLineEdit(self.settings.get("upload_status"))
                out_layout.addWidget(self.uploadEdit, 1, 1)
                
                out_layout.addWidget(QLabel("Scan Non-Media Files:"), 2, 0)
                self.scanOtherCheck = QCheckBox("Include other file types in scan")
                self.scanOtherCheck.setChecked(self.settings.get("scan_other_types", False))
                out_layout.addWidget(self.scanOtherCheck, 2, 1)
                
                out_layout.addWidget(QLabel("Recursive Scan:"), 3, 0)
                self.recursiveCheck = QCheckBox("Scan subdirectories")
                self.recursiveCheck.setChecked(self.settings.get("recursive_scan", True))
                out_layout.addWidget(self.recursiveCheck, 3, 1)
                
                out_layout.addWidget(QLabel("Safe Mode (Large Files):"), 4, 0)
                self.safeModeCheck = QCheckBox("Enable safe mode for large files")
                self.safeModeCheck.setChecked(self.settings.get("safe_mode", True))
                out_layout.addWidget(self.safeModeCheck, 4, 1)
                
                out_layout.addWidget(QLabel("Binary Search:"), 5, 0)
                self.binarySearchCheck = QCheckBox("Enable binary search for sequences (faster)")
                self.binarySearchCheck.setChecked(self.settings.get("binary_search", True))
                out_layout.addWidget(self.binarySearchCheck, 5, 1)
                
                out_group.setLayout(out_layout)
                
                # Predefined values settings
                predef_group = QGroupBox("Predefined Values")
                predef_layout = QGridLayout()
                
                # Submitted For values
                predef_layout.addWidget(QLabel("Submitted For Options:"), 0, 0)
                self.submittedForEdit = QLineEdit()
                self.submittedForEdit.setText(",".join(self.settings.get("custom_submitted_for", [])))
                predef_layout.addWidget(self.submittedForEdit, 0, 1)
                
                # Delivery Notes values
                predef_layout.addWidget(QLabel("Delivery Notes Options:"), 1, 0)
                self.deliveryNotesEdit = QLineEdit()
                self.deliveryNotesEdit.setText(",".join(self.settings.get("custom_delivery_notes", [])))
                predef_layout.addWidget(self.deliveryNotesEdit, 1, 1)
                
                predef_group.setLayout(predef_layout)
                
                # File type settings 
                filetype_group = QGroupBox("Filetype Settings")
                filetype_layout = QVBoxLayout()
                
                # Image types
                img_layout = QHBoxLayout()
                img_layout.addWidget(QLabel("Image Types:"))
                self.imgTypesEdit = QLineEdit(",".join(self.settings.get("image_types")))
                img_layout.addWidget(self.imgTypesEdit)
                filetype_layout.addLayout(img_layout)
                
                # Video types
                vid_layout = QHBoxLayout()
                vid_layout.addWidget(QLabel("Video Types:"))
                self.vidTypesEdit = QLineEdit(",".join(self.settings.get("video_types")))
                vid_layout.addWidget(self.vidTypesEdit)
                filetype_layout.addLayout(vid_layout)
                
                # Other types
                other_layout = QHBoxLayout()
                other_layout.addWidget(QLabel("Other Types:"))
                self.otherTypesEdit = QLineEdit(",".join(self.settings.get("other_types")))
                other_layout.addWidget(self.otherTypesEdit)
                filetype_layout.addLayout(other_layout)
                
                filetype_group.setLayout(filetype_layout)
                
                # Save/Cancel buttons
                btn_layout = QHBoxLayout()
                save_settings_btn = QPushButton("Save Settings")
                save_settings_btn.clicked.connect(self.save_settings)
                reset_settings_btn = QPushButton("Reset to Defaults")
                reset_settings_btn.clicked.connect(self.reset_settings)
                btn_layout.addWidget(save_settings_btn)
                btn_layout.addWidget(reset_settings_btn)
                
                # Assemble layout
                layout.addWidget(pat_group)
                layout.addWidget(out_group)
                layout.addWidget(predef_group)
                layout.addWidget(filetype_group)
                layout.addLayout(btn_layout)
                layout.addStretch()
                
                settings_tab.setLayout(layout)
                self.tab_widget.addTab(settings_tab, "Settings")
            
            except Exception as e:
                logger.error(f"Error creating settings tab: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("GUI Error", f"Error creating settings tab: {e}")
                
        def create_advanced_tab(self):
            """Create the advanced settings tab"""
            try:
                advanced_tab = QWidget()
                layout = QVBoxLayout()
                
                # Performance settings
                perf_group = QGroupBox("Performance Settings")
                perf_layout = QGridLayout()
                
                perf_layout.addWidget(QLabel("Worker Threads:"), 0, 0)
                self.workerThreads = QSpinBox()
                self.workerThreads.setRange(1, 16)
                self.workerThreads.setValue(self.settings.get("worker_threads", 8))
                self.workerThreads.setToolTip("Number of parallel processing threads")
                perf_layout.addWidget(self.workerThreads, 0, 1)
                
                perf_layout.addWidget(QLabel("Large File Threshold (MB):"), 1, 0)
                self.fileSizeWarning = QSpinBox()
                self.fileSizeWarning.setRange(100, 10000)
                self.fileSizeWarning.setValue(self.settings.get("file_size_warning", 500))
                self.fileSizeWarning.setSingleStep(100)
                self.fileSizeWarning.setToolTip("Files larger than this size will trigger special handling")
                perf_layout.addWidget(self.fileSizeWarning, 1, 1)
                
                perf_layout.addWidget(QLabel("Binary Search Depth:"), 2, 0)
                self.samplingDepth = QSpinBox()
                self.samplingDepth.setRange(3, 20)
                self.samplingDepth.setValue(self.settings.get("max_sampling_depth", 10))
                self.samplingDepth.setToolTip("Depth of binary search tree for sequence detection")
                perf_layout.addWidget(self.samplingDepth, 2, 1)
                
                perf_group.setLayout(perf_layout)
                
                # Binary search explanation
                binary_group = QGroupBox("Binary Search Algorithm")
                binary_layout = QVBoxLayout()
                
                binary_text = QLabel(
                    "Binary Search: When enabled, the application uses a highly optimized algorithm to detect sequence ranges:\n\n"
                    "• Instead of scanning every file, we use a binary tree approach\n"
                    "• First identifies potential sequence patterns using fast pattern matching\n"
                    "• Then uses binary search to quickly find start and end frames\n"
                    "• Automatically samples frames to identify gaps in sequences\n"
                    "• Can be 10-100x faster than scanning every file individually\n\n"
                    "This makes processing large directories with millions of files much faster."
                )
                binary_text.setWordWrap(True)
                
                binary_layout.addWidget(binary_text)
                binary_group.setLayout(binary_layout)
                
                # Optimizations info
                opt_group = QGroupBox("Performance Optimizations")
                opt_layout = QVBoxLayout()
                
                opt_text = QLabel(
                    "This version includes several major performance improvements:\n\n"
                    "• Binary search algorithm for sequence detection\n"
                    "• Hierarchical file pattern detection\n"
                    "• Optimized directory scanning using scandir()\n"
                    "• Multi-level caching system for metadata\n"
                    "• Reduced log verbosity for better performance\n"
                    "• Batch processing with priority-based sampling\n"
                    "• Fast filename validation with pattern caching\n\n"
                    "These improvements maintain stability while providing dramatically improved speed."
                )
                opt_text.setWordWrap(True)
                
                opt_layout.addWidget(opt_text)
                opt_group.setLayout(opt_layout)
                
                # Save button for advanced settings
                btn_layout = QHBoxLayout()
                save_advanced_btn = QPushButton("Save Advanced Settings")
                save_advanced_btn.clicked.connect(self.save_advanced_settings)
                btn_layout.addWidget(save_advanced_btn)
                
                # Assemble layout
                layout.addWidget(perf_group)
                layout.addWidget(binary_group)
                layout.addWidget(opt_group)
                layout.addLayout(btn_layout)
                layout.addStretch()
                
                advanced_tab.setLayout(layout)
                self.tab_widget.addTab(advanced_tab, "Advanced")
            
            except Exception as e:
                logger.error(f"Error creating advanced tab: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("GUI Error", f"Error creating advanced tab: {e}")

        def create_help_tab(self):
            """Create the help/about tab"""
            try:
                help_tab = QWidget()
                layout = QVBoxLayout()
                
                # About section
                about_group = QGroupBox("About")
                about_layout = QVBoxLayout()
                
                title_label = QLabel(f"File Scanner - CSV Creator v{VERSION}")
                title_label.setAlignment(Qt.AlignCenter)
                font = title_label.font()
                font.setBold(True)
                font.setPointSize(14)
                title_label.setFont(font)
                
                about_text = QLabel(
                    "This application scans folders for media files that match the specified naming pattern "
                    "and creates CSV reports for delivery packages.\n\n"
                    "The scanner can identify image sequences and process them accordingly.\n\n"
                    f"Version {VERSION} introduces a binary search algorithm for extremely fast sequence detection, "
                    "making it up to 10x faster than previous versions, particularly for directories with millions of files.\n\n"
                    "New in v0.21: Dedicated Results tab with improved multi-row editing capabilities."
                )
                about_text.setWordWrap(True)
                
                about_layout.addWidget(title_label)
                about_layout.addWidget(about_text)
                about_group.setLayout(about_layout)
                
                # Filename pattern help
                pattern_group = QGroupBox("Filename Pattern")
                pattern_layout = QVBoxLayout()
                
                pattern_text = QLabel(
                    "The application expects filenames to follow this pattern:\n\n"
                    "SEQID[SHOT]_DESCRIPTION_[PIXELMAPPING]RESOLUTION_COLORSPACE[_FPS]_vVERSION[_FRAME].EXTENSION\n\n"
                    "Where:\n"
                    "- SEQID is a sequence identifier (3-4 letters by default)\n"
                    "- SHOT is an optional 4-digit shot number (directly after SEQID, no underscore)\n"
                    "- DESCRIPTION is the content description (alphanumeric and hyphens)\n"
                    "- PIXELMAPPING is an optional LL180 or LL360 indicator (directly before RESOLUTION, no underscore)\n"
                    "- RESOLUTION is the resolution (digits followed by 'K')\n"
                    "- COLORSPACE is the color space and gamma\n"
                    "- FPS is an optional frames-per-second value\n"
                    "- VERSION is a version number (3 digits by default)\n"
                    "- FRAME is a frame number for image sequences\n\n"
                    "Example: ABC1234_CloudFormation_LL1804K_sRGB_24_v001.mov\n\n"
                    "For other file types (ZIP, PDF, etc.), the application will attempt to extract whatever "
                    "information it can and include them in the report with appropriate file size information."
                )
                pattern_text.setWordWrap(True)
                
                pattern_layout.addWidget(pattern_text)
                pattern_group.setLayout(pattern_layout)
                
                # Using Results Tab help
                results_group = QGroupBox("Using the Results Tab")
                results_layout = QVBoxLayout()
                
                results_text = QLabel(
                    "The Results Tab provides advanced editing features:\n\n"
                    "• Select multiple rows by clicking while holding Ctrl/Cmd or Shift\n"
                    "• Right-click on selected rows to edit specific fields for all selected rows at once\n"
                    "• Use the 'Edit Submitted For' and 'Edit Delivery Notes' buttons for quick access\n"
                    "• Custom values you enter will be remembered for future use\n"
                    "• The table is fully resizable - drag column borders to adjust width\n"
                    "• Sort the table by clicking on column headers\n\n"
                    "When finished editing, use the 'Save CSV from Results' button to save the final data."
                )
                results_text.setWordWrap(True)
                
                results_layout.addWidget(results_text)
                results_group.setLayout(results_layout)
                
                # Tips section
                tips_group = QGroupBox("Performance Tips")
                tips_layout = QVBoxLayout()
                
                tips_text = QLabel(
                    "Tips for optimal performance:\n\n"
                    "• Keep Binary Search enabled for maximum performance with large sequences\n"
                    "• For directories with millions of files, increase Binary Search Depth to 12-15\n"
                    "• Adjust worker threads based on your CPU cores (8 is optimal for most systems)\n"
                    "• Keep Safe Mode enabled for 12K+ EXR files\n"
                    "• When scanning large directories with many file types, select only the ones you need\n\n"
                    "The binary search algorithm makes this version much faster than previous versions "
                    "by intelligently detecting sequences without checking every file."
                )
                tips_text.setWordWrap(True)
                
                tips_layout.addWidget(tips_text)
                tips_group.setLayout(tips_layout)
                
                # Assemble layout
                layout.addWidget(about_group)
                layout.addWidget(pattern_group)
                layout.addWidget(results_group)
                layout.addWidget(tips_group)
                layout.addStretch()
                
                help_tab.setLayout(layout)
                self.tab_widget.addTab(help_tab, "Help")
            
            except Exception as e:
                logger.error(f"Error creating help tab: {e}")
                logger.error(traceback.format_exc())
                self.show_error_dialog("GUI Error", f"Error creating help tab: {e}")

        def add_directory(self):
            """Open file dialog to add a directory"""
            try:
                directory = QFileDialog.getExistingDirectory(
                    self, "Select Directory", os.path.expanduser("~")
                )
                if directory:
                    # Check if already in the list
                    for i in range(self.folder_list.count()):
                        if self.folder_list.item(i).text() == directory:
                            return
                    
                    it = QListWidgetItem(directory)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked)
                    self.folder_list.addItem(it)
            
            except Exception as e:
                logger.error(f"Error adding directory: {e}")
                logger.error(traceback.format_exc())
                self.status_bar.showMessage(f"Error adding directory: {e}", 3000)

        def remove_selected_folders(self):
            """Remove selected folders from the list"""
            try:
                for i in reversed(range(self.folder_list.count())):
                    it = self.folder_list.item(i)
                    if it.isSelected():
                        self.folder_list.takeItem(i)
            
            except Exception as e:
                logger.error(f"Error removing folders: {e}")
                logger.error(traceback.format_exc())
                self.status_bar.showMessage(f"Error removing folders: {e}", 3000)

        def clear_folders(self):
            """Clear all folders from the list"""
            try:
                self.folder_list.clear()
            except Exception as e:
                logger.error(f"Error clearing folders: {e}")
                logger.error(traceback.format_exc())
            
        def refresh_file_list(self):
            """Refresh the file list by clearing cache and rescanning"""
            try:
                if hasattr(self.processor, 'file_cache'):
                    self.processor.file_cache.clear()
                if hasattr(self.processor, 'dir_cache'):
                    self.processor.dir_cache.clear()
                if hasattr(self.processor, 'fast_validation_cache'):
                    self.processor.fast_validation_cache.clear()
                if hasattr(self.processor, 'sequence_cache'):
                    self.processor.sequence_cache.clear()
                
                # Clear the FileUtils cache too
                FileUtils.clear_caches()
                    
                self.log_window.appendPlainText("All caches cleared. Ready for a fresh scan.")
                self.status_bar.showMessage("Cache cleared. Click 'Start Scan' to rescan folders.", 5000)
            
            except Exception as e:
                logger.error(f"Error refreshing file list: {e}")
                logger.error(traceback.format_exc())
                self.status_bar.showMessage(f"Error refreshing file list: {e}", 3000)

        def on_select_all(self, state):
            """Handle 'Select All File Types' checkbox"""
            try:
                if state:
                    # Check all file type checkboxes
                    for cb in self.imageTypeCheckboxes.values():
                        cb.setChecked(True)
                    for cb in self.videoTypeCheckboxes.values():
                        cb.setChecked(True)
                    for cb in self.otherTypeCheckboxes.values():
                        cb.setChecked(True)
                        
                    # Uncheck the category selectors
                    self.select_img.blockSignals(True)
                    self.select_vid.blockSignals(True)
                    self.select_other.blockSignals(True)
                    
                    self.select_img.setChecked(False)
                    self.select_vid.setChecked(False)
                    self.select_other.setChecked(False)
                    
                    self.select_img.blockSignals(False)
                    self.select_vid.blockSignals(False)
                    self.select_other.blockSignals(False)
            
            except Exception as e:
                logger.error(f"Error in select all: {e}")
                logger.error(traceback.format_exc())

        def on_select_image(self, state):
            """Handle 'Select All Image Files' checkbox"""
            try:
                if state:
                    # Check image checkboxes, uncheck others
                    for cb in self.imageTypeCheckboxes.values():
                        cb.setChecked(True)
                    for cb in self.videoTypeCheckboxes.values():
                        cb.setChecked(False)
                    for cb in self.otherTypeCheckboxes.values():
                        cb.setChecked(False)
                        
                    # Uncheck other selectors
                    self.select_all.blockSignals(True)
                    self.select_vid.blockSignals(True)
                    self.select_other.blockSignals(True)
                    
                    self.select_all.setChecked(False)
                    self.select_vid.setChecked(False)
                    self.select_other.setChecked(False)
                    
                    self.select_all.blockSignals(False)
                    self.select_vid.blockSignals(False)
                    self.select_other.blockSignals(False)
            
            except Exception as e:
                logger.error(f"Error in select image: {e}")
                logger.error(traceback.format_exc())

        def on_select_video(self, state):
            """Handle 'Select All Video Files' checkbox"""
            try:
                if state:
                    # Check video checkboxes, uncheck others
                    for cb in self.videoTypeCheckboxes.values():
                        cb.setChecked(True)
                    for cb in self.imageTypeCheckboxes.values():
                        cb.setChecked(False)
                    for cb in self.otherTypeCheckboxes.values():
                        cb.setChecked(False)
                        
                    # Uncheck other selectors
                    self.select_all.blockSignals(True)
                    self.select_img.blockSignals(True)
                    self.select_other.blockSignals(True)
                    
                    self.select_all.setChecked(False)
                    self.select_img.setChecked(False)
                    self.select_other.setChecked(False)
                    
                    self.select_all.blockSignals(False)
                    self.select_img.blockSignals(False)
                    self.select_other.blockSignals(False)
            
            except Exception as e:
                logger.error(f"Error in select video: {e}")
                logger.error(traceback.format_exc())
                
        def on_select_other(self, state):
            """Handle 'Select All Other Files' checkbox"""
            try:
                if state:
                    # Check other checkboxes, uncheck media
                    for cb in self.otherTypeCheckboxes.values():
                        cb.setChecked(True)
                    for cb in self.imageTypeCheckboxes.values():
                        cb.setChecked(False)
                    for cb in self.videoTypeCheckboxes.values():
                        cb.setChecked(False)
                        
                    # Uncheck other selectors
                    self.select_all.blockSignals(True)
                    self.select_img.blockSignals(True)
                    self.select_vid.blockSignals(True)
                    
                    self.select_all.setChecked(False)
                    self.select_img.setChecked(False)
                    self.select_vid.setChecked(False)
                    
                    self.select_all.blockSignals(False)
                    self.select_img.blockSignals(False)
                    self.select_vid.blockSignals(False)
            
            except Exception as e:
                logger.error(f"Error in select other: {e}")
                logger.error(traceback.format_exc())

        def update_delivery_field(self):
            """Update the delivery package field in the results table"""
            try:
                txt = self.delEdit.text().strip()
                
                # Update all rows in the results table
                for r in range(self.results_table.rowCount()):
                    self.results_table.setItem(r, 9, QTableWidgetItem(txt))
            
            except Exception as e:
                logger.error(f"Error updating delivery field: {e}")
                logger.error(traceback.format_exc())

        def start_scan(self):
            """Start the file scanning process"""
            try:
                # Get selected directories
                dirs = [
                    self.folder_list.item(i).text()
                    for i in range(self.folder_list.count())
                    if self.folder_list.item(i).checkState() == Qt.Checked
                ]
                if not dirs:
                    QMessageBox.warning(self, "No Folders Selected", "Please select at least one folder.")
                    return

                # Get selected file types
                types = []
                
                # Collect image types
                for ext, cb in self.imageTypeCheckboxes.items():
                    if cb.isChecked():
                        types.append(ext)
                        
                # Collect video types
                for ext, cb in self.videoTypeCheckboxes.items():
                    if cb.isChecked():
                        types.append(ext)
                        
                # Determine if other files should be included
                include_other_types = any(cb.isChecked() for cb in self.otherTypeCheckboxes.values())
                
                # We need at least one selected file type (or other types enabled)
                if not types and not include_other_types:
                    QMessageBox.warning(self, "No File Types", "Please select at least one file type.")
                    return

                # Clear previous results
                self.results_table.setRowCount(0)
                self.progress_bar.setValue(0)
                self.log_window.clear()
                self.save_btn.setEnabled(False)

                # Get delivery package name
                delivery_package = self.delEdit.text().strip()
                if delivery_package:
                    # Save to settings
                    self.settings.set("last_delivery_package", delivery_package)
                    self.settings.save()

                # Update directory history
                last_dirs = self.settings.get("last_directories", [])
                new_dirs = []
                for d in dirs:
                    if d not in new_dirs:
                        new_dirs.append(d)
                # Keep only the last 10 directories
                self.settings.set("last_directories", new_dirs[:10])
                self.settings.save()

                # Add scanning message to log
                self.log_window.appendPlainText(f"Starting scan of {len(dirs)} directories...")
                self.log_window.appendPlainText(f"Recursive scan: {self.settings.get('recursive_scan', True)}")
                self.log_window.appendPlainText(f"Safe mode: {self.settings.get('safe_mode', True)}")
                self.log_window.appendPlainText(f"Binary search: {self.settings.get('binary_search', True)}")
                self.log_window.appendPlainText(f"Looking for {len(types)} file types" + 
                                              (" plus other file types" if include_other_types else ""))

                # Create and start the scanner thread
                self.scanner = FileScanner(
                    self.processor,
                    directories=dirs,
                    delivery_package=delivery_package,
                    file_types=types,
                    include_other_types=include_other_types
                )
                self.scanner.progress.connect(self.progress_bar.setValue)
                self.scanner.update_preview.connect(self.update_preview)
                self.scanner.log_message.connect(self.log_window.appendPlainText)
                self.scanner.scan_complete.connect(self.on_scan_complete)
                
                # Disable the scan button while running
                self.start_btn.setEnabled(False)
                self.status_bar.showMessage("Scanning files...")
                
                # Start the scan
                self.scanner.start()
            
            except Exception as e:
                logger.error(f"Error starting scan: {e}")
                logger.error(traceback.format_exc())
                self.log_window.appendPlainText(f"Error starting scan: {e}")
                self.start_btn.setEnabled(True)
                self.status_bar.showMessage(f"Error starting scan: {e}", 3000)

        def on_scan_complete(self, headers, rows):
            """Handle scan completion"""
            try:
                self.start_btn.setEnabled(True)
                self.save_btn.setEnabled(len(rows) > 0)
                
                # Update the results tab
                if len(rows) > 0:
                    # Enable the results tab
                    results_tab_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "", Qt.FindChildrenRecursively))
                    self.tab_widget.setTabEnabled(results_tab_index, True)
                    
                    # Switch to the results tab
                    self.tab_widget.setCurrentIndex(1)  # Results tab is at index 1
                    
                    # Show a message encouraging the user to review and edit
                    self.status_bar.showMessage(f"Scan complete. Found {len(rows)} items. You can now review and edit in the Results tab.", 5000)
                else:
                    self.status_bar.showMessage(f"Scan complete. No valid items found.", 3000)
            
            except Exception as e:
                logger.error(f"Error in scan completion handler: {e}")
                logger.error(traceback.format_exc())
                self.start_btn.setEnabled(True)
                self.status_bar.showMessage("Error handling scan completion", 3000)

        def update_preview(self, data):
            """Update the preview table with scan results"""
            try:
                if data['action'] == 'init':
                    # Clear the results table
                    self.results_table.setRowCount(0)
                    
                    # Make sure the results tab is enabled
                    results_index = 1  # Results tab is at index 1
                    self.tab_widget.setTabEnabled(results_index, True)
                    
                elif data['action'] == 'update':
                    # Add row to results table
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    for c, v in enumerate(data['row_data']):
                        itm = QTableWidgetItem(str(v))
                        itm.setFlags(itm.flags() | Qt.ItemIsEditable)
                        self.results_table.setItem(row, c, itm)
                        
                elif data['action'] == 'complete':
                    # Resize columns to content
                    self.results_table.resizeColumnsToContents()
                    
                    # Sort by shot name
                    if self.results_table.rowCount() > 0:
                        self.results_table.sortItems(1)
                    
                    # Enable the results tab
                    results_index = 1  # Results tab is at index 1
                    self.tab_widget.setTabEnabled(results_index, True)
            
            except Exception as e:
                logger.error(f"Error updating preview: {e}")
                logger.error(traceback.format_exc())
                self.log_window.appendPlainText(f"Error updating preview: {e}")

        def save_csv(self):
            """Save the table contents to a CSV file (redirects to save_csv_from_results)"""
            self.save_csv_from_results()
            
        def save_csv_from_results(self):
            """
            Save the results table contents to a CSV file.
            Enhanced to pre-fill the delivery package name in the save dialog.
            """
            try:
                if self.results_table.rowCount() == 0:
                    QMessageBox.information(self, "No Data", "No scan data to save.")
                    return
                    
                # Get column headers
                headers = []
                for c in range(self.results_table.columnCount()):
                    headers.append(self.results_table.horizontalHeaderItem(c).text())
                    
                # Get row data
                rows = []
                for r in range(self.results_table.rowCount()):
                    row = []
                    for c in range(len(headers)):
                        itm = self.results_table.item(r, c)
                        row.append(itm.text() if itm else "")
                    rows.append(row)
                
                # Get delivery package name to use as default filename
                delivery_package = self.delEdit.text().strip()
                if not delivery_package:
                    # Try to get it from the results table
                    for r in range(self.results_table.rowCount()):
                        item = self.results_table.item(r, 9)  # Column 9 is "Delivery Package Name"
                        if item and item.text().strip():
                            delivery_package = item.text().strip()
                            break
                    
                    # If still empty, use a generic name
                    if not delivery_package:
                        delivery_package = "delivery_package"
                
                # Sanitize filename - replace invalid characters
                safe_filename = re.sub(r'[\\/*?:"<>|]', "_", delivery_package)
                
                # Show save dialog with pre-filled filename
                fn, _ = QFileDialog.getSaveFileName(
                    self, "Save CSV File", 
                    safe_filename + ".csv",  # Pre-fill with delivery package name
                    "CSV Files (*.csv)"
                )
                
                if fn:
                    if not fn.lower().endswith('.csv'):
                        fn += '.csv'
                    if save_csv(fn, headers, rows):
                        self.status_bar.showMessage(f"CSV saved to {fn}", 5000)
                    else:
                        QMessageBox.critical(self, "Error", f"Failed to save CSV to {fn}")
            
            except Exception as e:
                logger.error(f"Error saving CSV: {e}")
                logger.error(traceback.format_exc())
                self.status_bar.showMessage(f"Error saving CSV: {e}", 3000)
        def save_settings(self):
            """Save current settings"""
            try:
                # Pattern settings
                self.settings.update({
                    "sequence_min": self.seqMin.value(),
                    "sequence_max": self.seqMax.value(),
                    "use_shot": self.shotCheck.isChecked(),
                    "use_pixel": self.pixCheck.isChecked(),
                    "use_fps": self.fpsCheck.isChecked(),
                    "version_min": self.verMin.value(),
                    "version_max": self.verMax.value(),
                    "sequence_threshold": self.seqThreshold.value(),
                    "vendor_name": self.vendorEdit.text(),
                    "upload_status": self.uploadEdit.text(),
                    "scan_other_types": self.scanOtherCheck.isChecked(),
                    "recursive_scan": self.recursiveCheck.isChecked(),
                    "safe_mode": self.safeModeCheck.isChecked(),
                    "binary_search": self.binarySearchCheck.isChecked(),
                    "custom_submitted_for": [x.strip() for x in self.submittedForEdit.text().split(',') if x.strip()],
                    "custom_delivery_notes": [x.strip() for x in self.deliveryNotesEdit.text().split(',') if x.strip()],
                    "image_types": [x.strip() for x in self.imgTypesEdit.text().split(',') if x.strip()],
                    "video_types": [x.strip() for x in self.vidTypesEdit.text().split(',') if x.strip()],
                    "other_types": [x.strip() for x in self.otherTypesEdit.text().split(',') if x.strip()]
                })
                
                # Update the processor with new settings
                self.processor.update_pattern()
                self.processor.image_types = self.settings.get("image_types")
                self.processor.video_types = self.settings.get("video_types")
                self.processor.other_types = self.settings.get("other_types")
                self.processor.safe_mode = self.settings.get("safe_mode")
                self.processor.binary_search = self.settings.get("binary_search")
                
                # Rebuild file type checkboxes if needed
                self.rebuild_filetype_checkboxes()
                
                QMessageBox.information(self, "Settings Saved", "Settings have been saved.")
            
            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
                
        def save_advanced_settings(self):
            """Save advanced settings"""
            try:
                # Advanced settings
                self.settings.update({
                    "worker_threads": self.workerThreads.value(),
                    "file_size_warning": self.fileSizeWarning.value(),
                    "max_sampling_depth": self.samplingDepth.value()
                })
                
                # Update processor settings
                self.processor.file_size_warning = self.settings.get("file_size_warning")
                self.processor.max_sampling_depth = self.settings.get("max_sampling_depth")
                
                QMessageBox.information(self, "Advanced Settings Saved", 
                                    "Advanced settings have been saved. They will take effect on the next scan.")
            
            except Exception as e:
                logger.error(f"Error saving advanced settings: {e}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"Failed to save advanced settings: {e}")
            
            except Exception as e:
                logger.error(f"Error saving advanced settings: {e}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"Failed to save advanced settings: {e}")

        def reset_settings(self):
            """Reset settings to defaults"""
            try:
                reply = QMessageBox.question(
                    self, "Reset Settings", 
                    "Are you sure you want to reset all settings to defaults?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.settings.data = DEFAULT_SETTINGS.copy()
                    self.settings.save()
                    self.load_from_settings()
                    self.rebuild_filetype_checkboxes()
                    QMessageBox.information(self, "Settings Reset", "Settings have been reset to defaults.")
            
            except Exception as e:
                logger.error(f"Error resetting settings: {e}")
                logger.error(traceback.format_exc())
                QMessageBox.critical(self, "Error", f"Failed to reset settings: {e}")

        def rebuild_filetype_checkboxes(self):
            """Rebuild file type checkboxes if file types have changed"""
            try:
                # Get current types
                current_img_types = set(self.imageTypeCheckboxes.keys())
                new_img_types = set(self.settings.get("image_types"))
                
                current_vid_types = set(self.videoTypeCheckboxes.keys())
                new_vid_types = set(self.settings.get("video_types"))
                
                current_other_types = set(self.otherTypeCheckboxes.keys())
                new_other_types = set(self.settings.get("other_types"))
                
                # Check if any changed
                if (current_img_types != new_img_types or 
                    current_vid_types != new_vid_types or
                    current_other_types != new_other_types):
                    # Need to rebuild UI in the scan tab
                    self.tab_widget.removeTab(0)  # Remove scan tab
                    self.create_scan_tab()  # Recreate scan tab
                    self.tab_widget.setCurrentIndex(0)  # Select scan tab
            
            except Exception as e:
                logger.error(f"Error rebuilding file type checkboxes: {e}")
                logger.error(traceback.format_exc())

        def load_from_settings(self):
            """Load UI state from settings"""
            try:
                # Set delivery package
                self.delEdit.setText(self.settings.get("last_delivery_package", ""))
                
                # Update settings tab values
                self.seqMin.setValue(self.settings.get("sequence_min"))
                self.seqMax.setValue(self.settings.get("sequence_max"))
                self.shotCheck.setChecked(self.settings.get("use_shot"))
                self.pixCheck.setChecked(self.settings.get("use_pixel"))
                self.fpsCheck.setChecked(self.settings.get("use_fps"))
                self.verMin.setValue(self.settings.get("version_min"))
                self.verMax.setValue(self.settings.get("version_max"))
                self.seqThreshold.setValue(self.settings.get("sequence_threshold"))
                self.vendorEdit.setText(self.settings.get("vendor_name"))
                self.uploadEdit.setText(self.settings.get("upload_status"))
                self.scanOtherCheck.setChecked(self.settings.get("scan_other_types", False))
                self.recursiveCheck.setChecked(self.settings.get("recursive_scan", True))
                self.safeModeCheck.setChecked(self.settings.get("safe_mode", True))
                self.binarySearchCheck.setChecked(self.settings.get("binary_search", True))
                
                # Set predefined values for drop-downs
                self.submittedForEdit.setText(",".join(self.settings.get("custom_submitted_for", [])))
                self.deliveryNotesEdit.setText(",".join(self.settings.get("custom_delivery_notes", [])))
                
                # File types
                self.imgTypesEdit.setText(",".join(self.settings.get("image_types")))
                self.vidTypesEdit.setText(",".join(self.settings.get("video_types")))
                self.otherTypesEdit.setText(",".join(self.settings.get("other_types")))
                
                # Load advanced settings
                if hasattr(self, 'workerThreads'):
                    self.workerThreads.setValue(self.settings.get("worker_threads", 8))
                if hasattr(self, 'fileSizeWarning'):
                    self.fileSizeWarning.setValue(self.settings.get("file_size_warning", 500))
                if hasattr(self, 'samplingDepth'):
                    self.samplingDepth.setValue(self.settings.get("max_sampling_depth", 10))
                
                # Load last used directories
                last_dirs = self.settings.get("last_directories", [])
                for d in last_dirs:
                    if os.path.isdir(d):
                        it = QListWidgetItem(d)
                        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                        it.setCheckState(Qt.Checked)
                        self.folder_list.addItem(it)
            
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
                logger.error(traceback.format_exc())

        def show_about(self):
            """Show about dialog"""
            try:
                QMessageBox.about(
                    self, 
                    f"About File Scanner v{VERSION}",
                    f"<h3>File Scanner v{VERSION}</h3>"
                    f"<p>A tool for scanning media files and creating CSV reports for delivery packages.</p>"
                    f"<p>New in this version:</p>"
                    f"<ul>"
                    f"<li>Dedicated results tab with improved resizability</li>"
                    f"<li>Multi-row editing for fast workflow</li>"
                    f"<li>Predefined value options for common fields</li>"
                    f"<li>Right-click context menu for advanced editing</li>"
                    f"<li>Binary search algorithm for supercharged performance</li>"
                    f"</ul>"
                    f"<p>Built with Python and PyQt5</p>"
                )
            except Exception as e:
                logger.error(f"Error showing about dialog: {e}")
                logger.error(traceback.format_exc())
            
        def show_debug_info(self):
            """Show debug information dialog"""
            try:
                debug_info = (
                    f"Version: {VERSION}\n"
                    f"Python Version: {sys.version}\n"
                    f"Platform: {sys.platform}\n"
                    f"Frozen: {FROZEN}\n"
                    f"GUI Available: {GUI_AVAILABLE}\n"
                    f"Log File: {log_file}\n"
                    f"Working Directory: {os.getcwd()}\n"
                    f"Settings File: {self.settings.settings_file}\n"
                    f"Safe Mode: {self.settings.get('safe_mode', True)}\n"
                    f"Binary Search: {self.settings.get('binary_search', True)}\n"
                    f"Worker Threads: {self.settings.get('worker_threads', 8)}\n"
                    f"Large File Threshold: {self.settings.get('file_size_warning', 500)} MB\n"
                    f"Binary Search Depth: {self.settings.get('max_sampling_depth', 10)}\n"
                    f"Custom Submitted For Values: {self.settings.get('custom_submitted_for', [])}\n"
                    f"Custom Delivery Notes: {self.settings.get('custom_delivery_notes', [])}\n"
                )
                
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Debug Information")
                msg_box.setText(debug_info)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.setDefaultButton(QMessageBox.Ok)
                msg_box.exec_()
            
            except Exception as e:
                logger.error(f"Error showing debug info: {e}")
                logger.error(traceback.format_exc())

        def closeEvent(self, event):
            """Handle window close event"""
            try:
                # Save window geometry
                if hasattr(self, 'settings'):
                    # self.settings.set("window_geometry", self.saveGeometry().hex())
                    geom_bytes = bytes(self.saveGeometry())
                    hex_str    = geom_bytes.hex()
                    self.settings.set("window_geometry", hex_str)
                    self.settings.save()
            except Exception as e:
                logger.error(f"Error saving settings during close: {e}")
                logger.error(traceback.format_exc())
            event.accept()

# --------------------
# Command-line interface
# --------------------
def run_cli(args):
    """Run in command-line mode"""
    logger.info(f"Starting File Scanner v{VERSION} in CLI mode")
    
    settings = Settings()
    processor = FileProcessor(settings)
    
    # Update settings from command line args if provided
    if args.sequence_min:
        settings.set("sequence_min", args.sequence_min)
    if args.sequence_max:
        settings.set("sequence_max", args.sequence_max)
    if args.version_min:
        settings.set("version_min", args.version_min)
    if args.version_max:
        settings.set("version_max", args.version_max)
    if args.sequence_threshold:
        settings.set("sequence_threshold", args.sequence_threshold)
    
    # Set use_shot, use_pixel, use_fps flags
    if args.no_shot:
        settings.set("use_shot", False)
    if args.no_pixel:
        settings.set("use_pixel", False)
    if args.no_fps:
        settings.set("use_fps", False)
    
    # Set recursive_scan flag
    if args.no_recursive:
        settings.set("recursive_scan", False)
        
    # Handle safe mode option
    if args.safe_mode is not None:
        settings.set("safe_mode", args.safe_mode)
    
    # Update worker threads
    if args.workers:
        settings.set("worker_threads", args.workers)
    
    # Update large file threshold
    if args.file_size_threshold:
        settings.set("file_size_warning", args.file_size_threshold)
    
    # Update binary search settings
    if args.binary_search is not None:
        settings.set("binary_search", args.binary_search)
    if args.sampling_depth:
        settings.set("max_sampling_depth", args.sampling_depth)
        
    # Update pattern with new settings
    processor.update_pattern()
    
    # Determine file types to include
    file_types = []
    if args.image_only:
        file_types.extend(settings.get("image_types"))
    elif args.video_only:
        file_types.extend(settings.get("video_types"))
    else:
        file_types.extend(settings.get("image_types"))
        file_types.extend(settings.get("video_types"))
    
    # Make sure we have directories to scan
    if not args.directories:
        logger.error("No directories specified for scanning")
        return 1
    
    # Check if directories exist
    valid_dirs = []
    for d in args.directories:
        if os.path.isdir(d):
            valid_dirs.append(d)
        else:
            logger.warning(f"Directory not found: {d}")
    
    if not valid_dirs:
        logger.error("No valid directories found")
        return 1
    
    # Define progress and log callbacks for CLI
    def progress_callback(value):
        progress = int(value / 10)
        sys.stdout.write('\r[' + '#' * progress + ' ' * (10 - progress) + f'] {value}%')
        sys.stdout.flush()
    
    def log_callback(msg):
        logger.info(msg)
    
    # Perform the scan
    logger.info(f"Scanning {len(valid_dirs)} directories for {len(file_types)} file types")
    logger.info(f"Recursive scan: {settings.get('recursive_scan', True)}")
    logger.info(f"Safe mode: {settings.get('safe_mode', True)}")
    logger.info(f"Binary search: {settings.get('binary_search', True)}")
    headers, rows = processor.scan_directories(
        valid_dirs,
        file_types,
        args.delivery or settings.get("last_delivery_package", "Default Package"),
        progress_callback,
        log_callback,
        None,
        args.include_other
    )
    
    # Print results summary
    print(f"\nScan complete. Found {len(rows)} valid items.")
    
    # Save CSV if output file specified
    if args.output:
        output_file = args.output
        if not output_file.lower().endswith('.csv'):
            output_file += '.csv'
        
        if save_csv(output_file, headers, rows):
            print(f"CSV saved to {output_file}")
        else:
            logger.error(f"Failed to save CSV to {output_file}")
            return 1
    
    return 0

def main():
    """Main entry point"""
    try:
        parser = argparse.ArgumentParser(description=f"File Scanner v{VERSION} - CSV Creator")
        parser.add_argument("--cli", action="store_true", help="Run in command-line mode")
        parser.add_argument("-d", "--directories", nargs="+", help="Directories to scan")
        parser.add_argument("-o", "--output", help="Output CSV file")
        parser.add_argument("--delivery", help="Delivery package name")
        parser.add_argument("--image-only", action="store_true", help="Only scan image files")
        parser.add_argument("--video-only", action="store_true", help="Only scan video files")
        parser.add_argument("--include-other", action="store_true", help="Include other file types")
        parser.add_argument("--no-recursive", action="store_true", help="Disable recursive directory scanning")
        parser.add_argument("--sequence-min", type=int, help="Minimum sequence ID length")
        parser.add_argument("--sequence-max", type=int, help="Maximum sequence ID length")
        parser.add_argument("--version-min", type=int, help="Minimum version digits")
        parser.add_argument("--version-max", type=int, help="Maximum version digits")
        parser.add_argument("--sequence-threshold", type=int, help="Minimum frames for sequence")
        parser.add_argument("--no-shot", action="store_true", help="Disable shot number in pattern")
        parser.add_argument("--no-pixel", action="store_true", help="Disable pixel mapping in pattern")
        parser.add_argument("--no-fps", action="store_true", help="Disable FPS in pattern")
        parser.add_argument("--safe-mode", type=bool, help="Enable/disable safe mode for large files")
        parser.add_argument("--workers", type=int, help="Number of worker threads")
        parser.add_argument("--file-size-threshold", type=int, help="File size threshold in MB")
        parser.add_argument("--binary-search", type=bool, help="Enable/disable binary search for sequences")
        parser.add_argument("--sampling-depth", type=int, help="Binary search sampling depth")
        
        args = parser.parse_args()
        
        # Run in CLI mode if requested or if directories are specified
        if args.cli or args.directories:
            return run_cli(args)
        
        # Run in GUI mode if Qt is available
        if GUI_AVAILABLE:
            # Create Qt application with proper error handling for bundled mode
            app = QApplication(sys.argv)
            
            # Set application info
            app.setApplicationName("CSV Creator")
            app.setApplicationVersion(VERSION)
            app.setOrganizationName("CG Fluids")
            
            # Create and show the main window
            window = MainWindow()
            window.show()
            
            # Start the event loop with proper exception handling
            try:
                return app.exec_()
            except Exception as e:
                logger.error(f"Error in Qt event loop: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.error("GUI libraries not available. Install with: pip install PyQt5 opencv-python")
            print("GUI mode requires PyQt5 and OpenCV. Install with: pip install PyQt5 opencv-python")
            return 1
            
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())