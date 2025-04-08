import os
import glob
import csv
from datetime import datetime
import sys
import re
import cv2  # For video duration
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
# New Naming Convention Parser
# --------------------
# Expected file name pattern (case-insensitive):
# <sequence><shotNumber>_<description>_<pixelMapping(optional)><resolution>_<colorspaceGamma>_(<fps>)?_v<version>[.<frame_padding>].<extension>
# The sequence group now accepts 3 or 4 letters.
new_pattern = re.compile(
    r'^(?P<sequence>[A-Za-z]{3,4})(?P<shotNumber>\d{4})_' +                 # sequence (3-4 letters) and shotNumber
    r'(?P<description>[^_]+)_' +                                             # description
    r'(?:(?P<pixelMapping>LL180|LL360))?(?P<resolution>\d+k|\d+K|\d+[a-zA-Z]+)_' +  # optional pixelMapping and resolution
    r'(?P<colorspaceGamma>[^_]+)' +                                          # colorspace+gamma field (ignored)
    r'(?:_(?P<fps>\d+))?_' +                                                 # optional underscore and fps
    r'v(?P<version>\d{1,4})' +                                               # version (with leading v)
    r'(?:(?:\.(?P<frame_padding>\d+))?\.(?P<extension>[^.]+))$',               # for images: .<frame_padding>.<extension>; for videos: .<extension>
    re.IGNORECASE
)

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
                    existing = [self.item(i).text() for i in range(self.count())]
                    if path not in existing:
                        item = QListWidgetItem(path)
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked)
                        self.addItem(item)
            event.acceptProposedAction()
        else:
            event.ignore()

# --------------------
# FileScanner Thread for New CSV Format
# --------------------
class FileScanner(QThread):
    progress = pyqtSignal(int)
    update_preview = pyqtSignal(dict)
    
    def __init__(self, directories, delivery_package, file_types):
        super().__init__()
        self.directories = directories  # List of directories to scan
        self.delivery_package = delivery_package
        self.file_types = file_types    # e.g. ["*.mp4", "*.exr", etc.]
        self.data = []
        self.paused = False

    def run(self):
        candidate_files = []
        for directory in self.directories:
            for ft in self.file_types:
                candidate_files.extend(glob.glob(os.path.join(directory, '**', ft), recursive=True))
        candidate_files = [f for f in candidate_files if new_pattern.match(os.path.basename(f))]
        total_files = len(candidate_files)
        processed_files = 0
        date = datetime.now().strftime("%m/%d/%Y")  # Delivery date in MM/DD/YYYY format
        headers = ["Name", "Shot Name", "Version Number", "Submitted For", "Notes", 
                   "File Type", "Resolution", "Duration", "Delivery Date", 
                   "Delivery Package Name", "Upload Status", "Vendor Name"]
        self.update_preview.emit({'action': 'init', 'headers': headers})
        sequences = {}  # Now use key = (name, extension) so that files with different ext are not skipped
        
        for f in candidate_files:
            if self.paused:
                return
            basename = os.path.basename(f)
            m = new_pattern.match(basename)
            if not m:
                continue
            groups = m.groupdict()
            # Column 1: Name — if frame_padding exists, remove it; otherwise full base name.
            if groups.get("frame_padding"):
                name = basename.rsplit('.', 2)[0]
            else:
                name = os.path.splitext(basename)[0]
            # Use a key that includes the extension (lower-case) to allow same name with different ext.
            key = (name, groups['extension'].lower())
            if key in sequences:
                continue
            sequences[key] = True

            # Column 2: Shot Name: <sequence>_<shotNumber>
            shot_name_field = f"{groups['sequence']}_{groups['shotNumber']}"
            # Column 3: Version Number: pad to 3 digits with leading "v"
            try:
                version_number = "v" + format(int(groups['version']), "03d")
            except Exception:
                version_number = "v" + groups['version']
            submitted_for = ""
            notes = ""
            file_type_field = groups['extension'].upper()
            resolution_field = groups['resolution']
            image_exts = ['EXR','JPG','TIFF','TIF','PNG','TGA','PSD']
            if file_type_field in image_exts:
                duration_field = "Still Frame"
            else:
                try:
                    cap = cv2.VideoCapture(f)
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration_field = str(frames)
                    cap.release()
                except Exception:
                    duration_field = ""
            delivery_date = date
            delivery_package_field = self.delivery_package
            upload_status_field = "Uploaded to Aspera"
            vendor_name_field = "CG Fluids"
            
            data_dict = {
                "Name": name,
                "Shot Name": shot_name_field,
                "Version Number": version_number,
                "Submitted For": submitted_for,
                "Notes": notes,
                "File Type": file_type_field,
                "Resolution": resolution_field,
                "Duration": duration_field,
                "Delivery Date": delivery_date,
                "Delivery Package Name": self.delivery_package,
                "Upload Status": upload_status_field,
                "Vendor Name": vendor_name_field
            }
            
            row_data = [
                data_dict["Name"],
                data_dict["Shot Name"],
                data_dict["Version Number"],
                data_dict["Submitted For"],
                data_dict["Notes"],
                data_dict["File Type"],
                data_dict["Resolution"],
                data_dict["Duration"],
                data_dict["Delivery Date"],
                data_dict["Delivery Package Name"],
                data_dict["Upload Status"],
                data_dict["Vendor Name"]
            ]
            
            self.data.append(data_dict)
            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100) if total_files > 0 else 0
            self.progress.emit(progress_percent)
            self.update_preview.emit({'action': 'update', 'row_data': row_data})
        
        self.progress.emit(100)
        self.update_preview.emit({'action': 'complete'})

# --------------------
# Main Window (Using Drag and Drop for Folder Selection)
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window size to approximately 975x800.
        self.setGeometry(100, 100, 975, 800)
        self.setWindowTitle("File Scanner - New CSV Format")
        self.scanner = None
        self.table = None
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

        # --- File Type Selection Section (Bordered) ---
        self.select_all_checkbox = QCheckBox("Select All File Types")
        self.select_all_checkbox.toggled.connect(self.on_select_all)
        select_all_layout = QHBoxLayout()
        select_all_layout.addWidget(self.select_all_checkbox, alignment=Qt.AlignCenter)

        self.select_image_checkbox = QCheckBox("Select All Image Files")
        self.select_image_checkbox.toggled.connect(self.on_select_image)
        self.select_video_checkbox = QCheckBox("Select All Video Types")
        self.select_video_checkbox.toggled.connect(self.on_select_video)
        image_types = ['exr', 'jpg', 'tiff', 'tif', 'png', 'tga', 'psd']
        video_types = ['mov', 'mxf', 'mp4']
        imageTypeInnerGroup = QGroupBox()
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
        
        imageGroupBox = QGroupBox("Image File Types")
        imageGroupLayout = QVBoxLayout()
        imageGroupLayout.addWidget(self.select_image_checkbox, alignment=Qt.AlignCenter)
        imageGroupLayout.addWidget(imageTypeInnerGroup)
        imageGroupBox.setLayout(imageGroupLayout)
        
        videoGroupBox = QGroupBox("Video File Types")
        videoGroupLayout = QVBoxLayout()
        videoGroupLayout.addWidget(self.select_video_checkbox, alignment=Qt.AlignCenter)
        videoGroupLayout.addWidget(videoTypeInnerGroup)
        videoGroupBox.setLayout(videoGroupLayout)
        
        fileTypesSplitLayout = QHBoxLayout()
        fileTypesSplitLayout.addWidget(imageGroupBox)
        fileTypesSplitLayout.addWidget(videoGroupBox)
        
        fileTypeSelectionGroup = QGroupBox("File Type Selection")
        fileTypeSelectionLayout = QVBoxLayout()
        fileTypeSelectionLayout.addLayout(select_all_layout)
        fileTypeSelectionLayout.addLayout(fileTypesSplitLayout)
        fileTypeSelectionGroup.setLayout(fileTypeSelectionLayout)

        # --- Delivery Package Name ---
        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(QLabel("Delivery Package Name:"))
        delivery_layout.addWidget(self.delivery_package)

        # --- Spacing before Preview Table ---
        spacing_before_preview = 20

        # --- Preview Table ---
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Name", "Shot Name", "Version Number", "Submitted For", "Notes",
            "File Type", "Resolution", "Duration", "Delivery Date",
            "Delivery Package Name", "Upload Status", "Vendor Name"
        ])

        # --- Spacing after Preview Table ---
        spacing_after_preview = 20

        # --- Control Buttons (Scan and Save) at the Bottom ---
        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.setMinimumSize(150, 80)
        self.start_scan_button.clicked.connect(self.start_scan)
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setMinimumSize(150, 80)
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
        main_layout.addWidget(folder_drop_group)
        main_layout.addWidget(self.remove_button)
        main_layout.addWidget(fileTypeSelectionGroup)
        main_layout.addLayout(delivery_layout)
        main_layout.addSpacing(spacing_before_preview)
        main_layout.addWidget(self.table)
        main_layout.addSpacing(spacing_after_preview)
        main_layout.addWidget(self.progress_bar)
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
    # Start scan: gather selected folders and file type patterns, then launch FileScanner.
    # --------------
    def start_scan(self):
        selected_dirs = self.get_checked_drop_folders()
        if not selected_dirs:
            QMessageBox.warning(self, "No Folders Selected", "Please drag and drop at least one folder and ensure it is checked.")
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
            self.table.setItem(row, 9, QTableWidgetItem(new_text))

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
                "Name", "Shot Name", "Version Number", "Submitted For", "Notes",
                "File Type", "Resolution", "Duration", "Delivery Date",
                "Delivery Package Name", "Upload Status", "Vendor Name"
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
            QMessageBox.information(self, "No Data", "There is no data to save. Please perform a scan first.")
            return

        updated_data = []
        headers = [
            "Name", "Shot Name", "Version Number", "Submitted For", "Notes",
            "File Type", "Resolution", "Duration", "Delivery Date",
            "Delivery Package Name", "Upload Status", "Vendor Name"
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
