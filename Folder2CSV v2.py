import os
import glob
import csv
from datetime import datetime
import sys
import re
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget, QLineEdit, QGroupBox, QCheckBox,
    QMessageBox, QListWidget, QListWidgetItem, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl

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
QLineEdit, QTableWidget, QListWidget {
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
# Custom QListWidget for Drag and Drop of Folders
# --------------------
class FolderDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    # Avoid duplicates.
                    existing = [self.item(i).text() for i in range(self.count())]
                    if path not in existing:
                        item = QListWidgetItem(path)
                        # Make the item checkable and default it to Checked.
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked)
                        self.addItem(item)
            event.acceptProposedAction()
        else:
            event.ignore()

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

                    # Check for sequences.
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
                        formatted_shot_name = base_name
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
# Main Window (Using Only Drag and Drop for Folder Selection)
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Make the window 40% smaller in width than 1625 → approximately 975, keep height at 800.
        self.setGeometry(100, 100, 975, 800)
        self.setWindowTitle("File Scanner")
        self.scanner = None
        self.table = None
        # Dictionaries for the grouped file type checkboxes.
        self.imageTypeCheckboxes = {}
        self.videoTypeCheckboxes = {}

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Drag and drop the folders you want to scan into the area below.")

        # --- Folder Drag-and-Drop Panel ---
        folder_drop_group = QGroupBox("Drag and Drop Folders (they will be checked by default)")
        self.folder_drop_list = FolderDropListWidget()
        self.folder_drop_list.setMinimumHeight(150)
        drop_layout = QVBoxLayout()
        drop_layout.addWidget(self.folder_drop_list)
        folder_drop_group.setLayout(drop_layout)

        # --- Remove Selected Folders Button ---
        self.remove_button = QPushButton("Remove Selected Folders")
        self.remove_button.setMinimumSize(150, 40)
        self.remove_button.clicked.connect(self.remove_selected_folders)

        # --- File Type Selection Section ---
        # Global select all for file types (this will be placed inside a bordered group)
        self.select_all_checkbox = QCheckBox("Select All File Types")
        self.select_all_checkbox.toggled.connect(self.on_select_all)
        select_all_layout = QHBoxLayout()
        select_all_layout.addWidget(self.select_all_checkbox, alignment=Qt.AlignCenter)

        # Create group boxes for image and video file types.
        self.select_image_checkbox = QCheckBox("Select All Image Files")
        self.select_image_checkbox.toggled.connect(self.on_select_image)
        self.select_video_checkbox = QCheckBox("Select All Video Types")
        self.select_video_checkbox.toggled.connect(self.on_select_video)
        image_types = ['exr', 'jpg', 'tiff', 'tif', 'png', 'tga', 'psd']
        video_types = ['mov', 'mxf', 'mp4']
        # Individual file type checkboxes (already bordered in a group box)
        imageTypeInnerGroup = QGroupBox()  # inner container for checkboxes (no title)
        imageLayout = QHBoxLayout()
        for ft in image_types:
            cb = QCheckBox(ft.upper())
            cb.setChecked(True)
            imageLayout.addWidget(cb)
            self.imageTypeCheckboxes[ft] = cb
        imageTypeInnerGroup.setLayout(imageLayout)

        videoTypeInnerGroup = QGroupBox()
        videoLayout = QHBoxLayout()
        for ft in video_types:
            cb = QCheckBox(ft.upper())
            cb.setChecked(True)
            videoLayout.addWidget(cb)
            self.videoTypeCheckboxes[ft] = cb
        videoTypeInnerGroup.setLayout(videoLayout)

        # Wrap the image section in its own group box.
        imageGroupBox = QGroupBox("Image File Types")
        imageGroupLayout = QVBoxLayout()
        imageGroupLayout.addWidget(self.select_image_checkbox, alignment=Qt.AlignCenter)
        imageGroupLayout.addWidget(imageTypeInnerGroup)
        imageGroupBox.setLayout(imageGroupLayout)

        # Wrap the video section in its own group box.
        videoGroupBox = QGroupBox("Video File Types")
        videoGroupLayout = QVBoxLayout()
        videoGroupLayout.addWidget(self.select_video_checkbox, alignment=Qt.AlignCenter)
        videoGroupLayout.addWidget(videoTypeInnerGroup)
        videoGroupBox.setLayout(videoGroupLayout)

        # Now group the entire file type selection in a parent group box.
        fileTypeSelectionGroup = QGroupBox("File Type Selection")
        fileTypeSelectionLayout = QVBoxLayout()
        fileTypeSelectionLayout.addLayout(select_all_layout)
        fileTypesSplitLayout = QHBoxLayout()
        fileTypesSplitLayout.addWidget(imageGroupBox)
        fileTypesSplitLayout.addWidget(videoGroupBox)
        fileTypeSelectionLayout.addLayout(fileTypesSplitLayout)
        fileTypeSelectionGroup.setLayout(fileTypeSelectionLayout)

        # --- Delivery Package Name ---
        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(QLabel("Delivery Package Name:"))
        delivery_layout.addWidget(self.delivery_package)

        # --- Spacing before Preview ---
        spacing_before_preview = 20

        # --- Preview Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Shot Name", "Version Number", "File Type", "File Size",
            "Delivery Date", "Delivery Package Name", "Upload Status",
            "Submission Notes"
        ])

        # --- Spacing after Preview (if needed) ---
        spacing_after_preview = 20

        # --- Control Buttons (Scan and Save) at the Bottom ---
        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.setMinimumSize(150, 80)  # 200% larger in height
        self.start_scan_button.clicked.connect(self.start_scan)
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setMinimumSize(150, 80)  # 200% larger in height
        self.save_csv_button.clicked.connect(self.save_csv)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_scan_button)
        button_layout.addWidget(self.save_csv_button)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        # --- Main Layout Assembly ---
        main_widget = QWidget(self)
        main_layout = QVBoxLayout(main_widget)
        # Top selections: folder drop area and remove button
        main_layout.addWidget(folder_drop_group)
        main_layout.addWidget(self.remove_button)
        # File type selection section with border
        main_layout.addWidget(fileTypeSelectionGroup)
        # Delivery package field
        main_layout.addLayout(delivery_layout)
        # Spacing before preview table
        main_layout.addSpacing(spacing_before_preview)
        # Preview table
        main_layout.addWidget(self.table)
        # Spacing after preview table
        main_layout.addSpacing(spacing_after_preview)
        # Progress bar
        main_layout.addWidget(self.progress_bar)
        # Buttons at the very bottom
        main_layout.addLayout(button_layout)
        self.setCentralWidget(main_widget)

    # --------------
    # Remove selected items from the drop list.
    # --------------
    def remove_selected_folders(self):
        for item in self.folder_drop_list.selectedItems():
            self.folder_drop_list.takeItem(self.folder_drop_list.row(item))

    # --------------
    # Collect checked folders from the drop list.
    # --------------
    def get_checked_drop_folders(self):
        checked = []
        for i in range(self.folder_drop_list.count()):
            item = self.folder_drop_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    # --------------
    # Start scan: gather selected folders (from the drop list) and file type patterns, then launch FileScanner.
    # --------------
    def start_scan(self):
        selected_dirs = self.get_checked_drop_folders()
        if not selected_dirs:
            QMessageBox.warning(
                self, "No Folders Selected",
                "Please drag and drop at least one folder and ensure it is checked."
            )
            return

        selected_types = []
        for ext, cb in self.imageTypeCheckboxes.items():
            if cb.isChecked():
                selected_types.append(f"*.{ext}")
        for ext, cb in self.videoTypeCheckboxes.items():
            if cb.isChecked():
                selected_types.append(f"*.{ext}")
        if not selected_types:
            QMessageBox.warning(
                self, "No File Types",
                "Please select at least one file type to scan."
            )
            return

        self.table.clearContents()
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Scanning selected folders...")

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
    # Make cells editable so that changes propagate to CSV.
    # --------------
    def update_preview(self, data):
        if data['action'] == 'init':
            pass
        elif data['action'] == 'update':
            row_data = data['row_data']
            row_count = self.table.rowCount()
            self.table.insertRow(row_count)
            headers = [
                "Shot Name", "Version Number", "File Type", "File Size",
                "Delivery Date", "Delivery Package Name", "Upload Status",
                "Submission Notes"
            ]
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
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
    # Before saving, update the scanner's data from the table's current content.
    # --------------
    def save_csv(self):
        if not self.scanner or not self.scanner.data:
            QMessageBox.information(
                self, "No Data", "There is no data to save. Please perform a scan first."
            )
            return

        updated_data = []
        headers = [
            "Shot Name", "Version Number", "File Type", "File Size",
            "Delivery Date", "Delivery Package Name", "Upload Status",
            "Submission Notes"
        ]
        row_count = self.table.rowCount()
        for i in range(row_count):
            row_dict = {}
            for j, header in enumerate(headers):
                item = self.table.item(i, j)
                text = item.text() if item is not None else ""
                row_dict[header] = text
            updated_data.append(row_dict)
        self.scanner.data = updated_data

        delivery_name = self.delivery_package.text().strip()
        for entry in self.scanner.data:
            entry["Delivery Package Name"] = delivery_name

        file_name, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "*.csv")
        if file_name:
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for entry in self.scanner.data:
                    row = [entry[header] for header in headers]
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
