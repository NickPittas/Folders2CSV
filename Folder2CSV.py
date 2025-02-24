import os
import glob
import csv
from datetime import datetime
import sys
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QTableWidget, QTableWidgetItem, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QProgressBar, QWidget, QLineEdit, QGroupBox,
                             QCheckBox, QMessageBox, QTreeWidget, QTreeWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --------------------
# Dark Theme Style Sheet
# --------------------
dark_stylesheet = """
QWidget {
    background-color: #2d2d30;
    color: #ffffff;
    font-family: Arial;
}
QPushButton {
    background-color: #3e3e42;
    border: 1px solid #565656;
    padding: 5px;
}
QPushButton:hover {
    background-color: #46464b;
}
QLineEdit, QTableWidget, QTreeWidget {
    background-color: #3e3e42;
    color: #ffffff;
    border: 1px solid #565656;
}
QGroupBox {
    border: 1px solid #565656;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 3px;
}
QCheckBox {
    spacing: 5px;
}
QProgressBar {
    border: 1px solid #565656;
    text-align: center;
    background-color: #3e3e42;
}
QProgressBar::chunk {
    background-color: #007acc;
}
QHeaderView::section {
    background-color: #3e3e42;
    padding: 4px;
    border: 1px solid #565656;
}
"""

# --------------------
# Helper Function to Extract Version Numbers
# --------------------
def get_version_from_filename(filename):
    match = re.search(r'([vV])(\d{1,4})', filename)
    if match:
        return f"v{match.group(2).zfill(4)}"
    else:
        return "None"

# --------------------
# FileScanner Thread (Now Accepts a List of Directories)
# --------------------
class FileScanner(QThread):
    progress = pyqtSignal(int)
    update_preview = pyqtSignal(dict)

    def __init__(self, directories, delivery_package, file_types):
        super().__init__()
        self.directories = directories  # List of directories to scan
        self.delivery_package = delivery_package
        self.file_types = file_types    # e.g. ["*.exr", "*.jpg", ...]
        self.data = []
        self.paused = False

    def run(self):
        total_files = 0
        processed_files = 0
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headers = ["Shot Name", "Version Number", "File Type",
                   "File Size (bytes)", "Delivery Date",
                   "Delivery Package Name", "Upload Status",
                   "Submission Notes"]
        self.update_preview.emit({'action': 'init', 'headers': headers})
        sequences = {}  # To ensure each sequence (or individual file) is processed only once.

        # Count total files across all selected directories and file type patterns.
        for directory in self.directories:
            for ft in self.file_types:
                pattern = os.path.join(directory, '**', ft)
                files = glob.glob(pattern, recursive=True)
                total_files += len(files)

        # Process each directory and file type.
        for directory in self.directories:
            for ft in self.file_types:
                pattern = os.path.join(directory, '**', ft)
                files = glob.glob(pattern, recursive=True)
                for f in files:
                    if self.paused:
                        return
                    filename = os.path.basename(f)
                    shot_name, ext = os.path.splitext(filename)
                    file_type = ext[1:]
                    version = get_version_from_filename(filename)

                    # Check for sequences: if the filename ends with a separator (space, underscore, or dot)
                    # followed by 4–10 digits, treat it as part of a sequence.
                    match_seq = re.search(r'^(.*?)([ _\.])(\d{4,10})$', shot_name)
                    if match_seq:
                        base_name = match_seq.group(1)
                        sep = match_seq.group(2)
                        seq_digits = match_seq.group(3)
                        seq_key = (base_name, sep, file_type)
                        if seq_key in sequences:
                            continue
                        pattern_seq = os.path.join(
                            os.path.dirname(f),
                            f"{base_name}{sep}" + ("[0-9]" * len(seq_digits)) + f".{file_type}"
                        )
                        seq_files = glob.glob(pattern_seq)
                        file_size = sum(os.path.getsize(fp) for fp in seq_files)
                        formatted_shot_name = base_name  # Only display base name.
                    else:
                        file_size = os.path.getsize(f)
                        formatted_shot_name = filename
                        seq_key = os.path.basename(f)
                    if seq_key in sequences:
                        continue
                    sequences[seq_key] = True

                    data_row = [
                        formatted_shot_name,
                        version if version != 'None' else "",
                        file_type.upper(),
                        f"{file_size} bytes",
                        date,
                        self.delivery_package,
                        "Uploaded to Aspera",
                        ""
                    ]
                    self.data.append({
                        "Shot Name": data_row[0],
                        "Version Number": data_row[1],
                        "File Type": data_row[2],
                        "File Size": file_size,
                        "Delivery Date": date,
                        "Delivery Package Name": self.delivery_package,
                        "Upload Status": "Uploaded to Aspera",
                        "Submission Notes": ""
                    })
                    processed_files += 1
                    progress_percent = int((processed_files / total_files) * 100) if total_files > 0 else 0
                    self.progress.emit(progress_percent)
                    self.update_preview.emit({'action': 'update', 'row_data': data_row})

        self.progress.emit(100)
        self.update_preview.emit({'action': 'complete'})

# --------------------
# Main Window
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Increase the window width by 25% (from 1300 to 1625) and set the height to 800.
        self.setGeometry(100, 100, 1625, 800)
        self.setWindowTitle("File Scanner")
        self.directory_path = ""
        self.scanner = None
        self.table = None
        # Dictionaries for the grouped file type checkboxes.
        self.imageTypeCheckboxes = {}
        self.videoTypeCheckboxes = {}

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Select a root folder and options to begin.")

        # --- Top Row: Root Directory Selection ---
        self.browse_button = QPushButton("Browse Directory")
        self.browse_button.setMinimumSize(150, 40)
        self.browse_button.clicked.connect(self.browse_directory)
        self.dir_lineedit = QLineEdit()
        self.dir_lineedit.setPlaceholderText("No directory selected")
        self.dir_lineedit.setReadOnly(True)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.browse_button)
        top_layout.addWidget(QLabel("Root Directory:"))
        top_layout.addWidget(self.dir_lineedit)

        # --- Subfolder Selection Panel (Tree View) ---
        self.folder_tree_widget = QTreeWidget()
        self.folder_tree_widget.setHeaderLabel("Folders")
        self.folder_tree_widget.setMinimumHeight(200)
        folder_group_box = QGroupBox("Select Subfolders to Scan (default: none selected)")
        folder_layout = QVBoxLayout()
        folder_layout.addWidget(self.folder_tree_widget)
        folder_group_box.setLayout(folder_layout)

        # --- Global Select-All Option for File Types ---
        self.select_all_checkbox = QCheckBox("Select All File Types")
        self.select_all_checkbox.toggled.connect(self.on_select_all)
        select_all_layout = QHBoxLayout()
        select_all_layout.addWidget(self.select_all_checkbox, alignment=Qt.AlignCenter)

        # --- Grouped File Type Selection ---
        self.select_image_checkbox = QCheckBox("Select All Image Files")
        self.select_image_checkbox.toggled.connect(self.on_select_image)
        self.select_video_checkbox = QCheckBox("Select All Video Types")
        self.select_video_checkbox.toggled.connect(self.on_select_video)
        image_types = ['exr', 'jpg', 'tiff', 'tif', 'png', 'tga', 'psd']
        video_types = ['mov', 'mxf', 'mp4']
        self.imageTypeGroupBox = QGroupBox("Image File Types")
        imageLayout = QHBoxLayout()
        for ft in image_types:
            cb = QCheckBox(ft.upper())
            cb.setChecked(True)
            imageLayout.addWidget(cb)
            self.imageTypeCheckboxes[ft] = cb
        self.imageTypeGroupBox.setLayout(imageLayout)
        self.videoTypeGroupBox = QGroupBox("Video File Types")
        videoLayout = QHBoxLayout()
        for ft in video_types:
            cb = QCheckBox(ft.upper())
            cb.setChecked(True)
            videoLayout.addWidget(cb)
            self.videoTypeCheckboxes[ft] = cb
        self.videoTypeGroupBox.setLayout(videoLayout)
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.select_image_checkbox, alignment=Qt.AlignCenter)
        leftLayout.addWidget(self.imageTypeGroupBox)
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.select_video_checkbox, alignment=Qt.AlignCenter)
        rightLayout.addWidget(self.videoTypeGroupBox)
        fileTypesSplitLayout = QHBoxLayout()
        fileTypesSplitLayout.addLayout(leftLayout)
        fileTypesSplitLayout.addLayout(rightLayout)

        # --- Delivery Package Name ---
        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(QLabel("Delivery Package Name:"))
        delivery_layout.addWidget(self.delivery_package)

        # --- Control Buttons (Scan and Save) ---
        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.setMinimumSize(150, 80)  # 200% larger in height
        self.start_scan_button.clicked.connect(self.start_scan)
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setMinimumSize(150, 80)  # 200% larger in height
        self.save_csv_button.clicked.connect(self.save_csv)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_scan_button)
        button_layout.addWidget(self.save_csv_button)

        # --- Preview Table and Progress Bar ---
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Shot Name", "Version Number", "File Type", "File Size",
                                               "Delivery Date", "Delivery Package Name", "Upload Status",
                                               "Submission Notes"])
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        # --- Main Layout Assembly ---
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(folder_group_box)
        main_layout.addLayout(select_all_layout)
        main_layout.addLayout(fileTypesSplitLayout)
        main_layout.addLayout(delivery_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.progress_bar)
        self.setCentralWidget(main_widget)

    # --------------
    # Recursively populate the tree widget with subfolders.
    # Each item stores its full path in UserRole.
    # --------------
    def populate_tree(self, parent_item, folder_path):
        try:
            for entry in sorted(os.listdir(folder_path)):
                full_path = os.path.join(folder_path, entry)
                if os.path.isdir(full_path):
                    child_item = QTreeWidgetItem(parent_item)
                    child_item.setText(0, entry)
                    child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                    child_item.setCheckState(0, Qt.Unchecked)
                    child_item.setData(0, Qt.UserRole, full_path)
                    # Recursively add subdirectories.
                    self.populate_tree(child_item, full_path)
        except Exception as e:
            pass

    # --------------
    # When the user selects a root directory, update the tree view.
    # --------------
    def update_folder_tree(self):
        self.folder_tree_widget.clear()
        root_item = QTreeWidgetItem(self.folder_tree_widget)
        root_item.setText(0, os.path.basename(self.directory_path))
        root_item.setFlags(root_item.flags() | Qt.ItemIsUserCheckable)
        root_item.setCheckState(0, Qt.Unchecked)
        root_item.setData(0, Qt.UserRole, self.directory_path)
        self.populate_tree(root_item, self.directory_path)
        self.folder_tree_widget.expandAll()

    # --------------
    # Root directory selection.
    # --------------
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Root Directory")
        if directory:
            self.directory_path = directory
            self.dir_lineedit.setText(directory)
            self.status_bar.showMessage("Root directory selected. Now select subfolders to scan.")
            self.update_folder_tree()

    # --------------
    # Recursively collect checked folders from the tree widget.
    # --------------
    def get_checked_folders(self, parent_item=None):
        checked = []
        if parent_item is None:
            # Top-level items
            top_count = self.folder_tree_widget.topLevelItemCount()
            for i in range(top_count):
                item = self.folder_tree_widget.topLevelItem(i)
                checked.extend(self.get_checked_folders(item))
        else:
            if parent_item.checkState(0) == Qt.Checked:
                full_path = parent_item.data(0, Qt.UserRole)
                checked.append(full_path)
            # Process children
            child_count = parent_item.childCount()
            for i in range(child_count):
                child = parent_item.child(i)
                checked.extend(self.get_checked_folders(child))
        return checked

    # --------------
    # Start scan: gather selected subfolders and file type patterns, then launch FileScanner.
    # --------------
    def start_scan(self):
        if not self.directory_path:
            QMessageBox.warning(self, "No Directory", "Please select a root directory first.")
            return

        selected_dirs = self.get_checked_folders()
        if not selected_dirs:
            QMessageBox.warning(self, "No Subfolders Selected", "Please select at least one subfolder to scan.")
            return

        selected_types = []
        for ext, cb in self.imageTypeCheckboxes.items():
            if cb.isChecked():
                selected_types.append(f"*.{ext}")
        for ext, cb in self.videoTypeCheckboxes.items():
            if cb.isChecked():
                selected_types.append(f"*.{ext}")
        if not selected_types:
            QMessageBox.warning(self, "No File Types", "Please select at least one file type to scan.")
            return

        self.table.clearContents()
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Scanning selected subfolders...")

        delivery_name = self.delivery_package.text().strip()
        self.scanner = FileScanner(selected_dirs, delivery_name, selected_types)
        self.scanner.progress.connect(self.update_progress)
        self.scanner.update_preview.connect(self.update_preview)
        self.scanner.start()

    # --------------
    # Update Delivery Package Name in UI and scanner.
    # --------------
    def update_delivery_field(self):
        new_text = self.delivery_package.text().strip()
        if self.scanner:
            self.scanner.delivery_package = new_text
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 5, QTableWidgetItem(new_text))

    # --------------
    # Update preview table as files are processed.
    # --------------
    def update_preview(self, data):
        if data['action'] == 'init':
            pass
        elif data['action'] == 'update':
            row_data = data['row_data']
            row_count = self.table.rowCount()
            self.table.insertRow(row_count)
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.table.setItem(row_count, col, item)
        elif data['action'] == 'complete':
            self.status_bar.showMessage("Scan complete.")

    # --------------
    # Update progress bar.
    # --------------
    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    # --------------
    # Save CSV file.
    # --------------
    def save_csv(self):
        if not self.scanner or not self.scanner.data:
            QMessageBox.information(self, "No Data", "There is no data to save. Please perform a scan first.")
            return
        delivery_name = self.delivery_package.text().strip()
        for entry in self.scanner.data:
            entry["Delivery Package Name"] = delivery_name
        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "*.csv")
        if file_name:
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = ["Shot Name", "Version Number", "File Type", "File Size",
                           "Delivery Date", "Delivery Package Name", "Upload Status",
                           "Submission Notes"]
                writer.writerow(headers)
                for entry in self.scanner.data:
                    row = [
                        entry["Shot Name"],
                        entry["Version Number"],
                        entry["File Type"],
                        str(entry["File Size"]),
                        entry["Delivery Date"],
                        entry["Delivery Package Name"],
                        entry["Upload Status"],
                        entry["Submission Notes"]
                    ]
                    writer.writerow(row)
            self.status_bar.showMessage("CSV saved successfully!")

    # --------------
    # Select-all slots for file type checkboxes.
    # --------------
    def on_select_all(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values():
                cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values():
                cb.setChecked(True)
            self.select_image_checkbox.blockSignals(True)
            self.select_video_checkbox.blockSignals(True)
            self.select_image_checkbox.setChecked(False)
            self.select_video_checkbox.setChecked(False)
            self.select_image_checkbox.blockSignals(False)
            self.select_video_checkbox.blockSignals(False)

    def on_select_video(self, state):
        if state:
            for cb in self.videoTypeCheckboxes.values():
                cb.setChecked(True)
            for cb in self.imageTypeCheckboxes.values():
                cb.setChecked(False)
            self.select_all_checkbox.blockSignals(True)
            self.select_image_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_image_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            self.select_image_checkbox.blockSignals(False)

    def on_select_image(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values():
                cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values():
                cb.setChecked(False)
            self.select_all_checkbox.blockSignals(True)
            self.select_video_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_video_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            self.select_video_checkbox.blockSignals(False)

# --------------------
# Main Application
# --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
