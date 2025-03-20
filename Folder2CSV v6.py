import os
import glob
import csv
from datetime import datetime
import sys
import re
import cv2  # For video duration
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget, QLineEdit, QGroupBox, QCheckBox,
    QMessageBox, QListWidget, QListWidgetItem, QFileDialog, QPlainTextEdit
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
QLineEdit, QTableWidget, QListWidget, QPlainTextEdit {
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
# Expected Naming Requirements (for reference)
# --------------------
requirements_str = (
    "Naming requirements:\n"
    "- The file name must start with exactly 3-4 letters followed by exactly 4 digits.\n"
    "- Then an underscore and a description (letters, digits, or hyphens).\n"
    "- Then an underscore and an optional pixel mapping (LL180 or LL360) immediately followed by resolution (digits followed by 'k' or 'K').\n"
    "- Then an underscore and a colorspace+gamma field (no underscores allowed).\n"
    "- Then an optional underscore and fps (digits) may appear (mandatory for sequences and video files).\n"
    "- Then an underscore, then 'v' followed by 1-4 digits (the version).\n"
    "- For image sequences, the frame padding (e.g. 001 or 0001) must appear and may be preceded by an underscore or a dot before the final dot and extension.\n"
    "- For video files, a dot and extension follow.\n"
)

# --------------------
# Validate a file name against expected criteria.
# Returns (errors, warnings) lists with only the specific issues.
# --------------------
def validate_filename(basename):
    errors = []
    warnings = []
    if '.' not in basename:
        errors.append("Missing file extension")
        return errors, warnings
    m = new_pattern.match(basename)
    if not m:
        errors.append("Filename does not match expected pattern")
        return errors, warnings
    groups = m.groupdict()
    token0 = groups['sequence'] + groups['shotNumber']
    if not re.match(r'^[A-Za-z]{3,4}\d{4}$', token0):
        errors.append("Token0 (sequence+shotNumber) must be 3-4 letters followed by 4 digits")
    if not groups['description']:
        errors.append("Description is missing")
    if not re.match(r'^\d+[kK]$', groups['resolution']):
        errors.append("Resolution must be digits followed by 'k' or 'K'")
    if not groups['colorspaceGamma']:
        errors.append("ColorspaceGamma field is missing")
    image_exts = ['exr','jpg','tiff','tif','png','tga','psd']
    ext = groups.get('extension', '').lower()
    if ext in image_exts:
        # For image sequences (with frame_padding), fps is mandatory.
        if groups.get('frame_padding'):
            if not groups.get('fps'):
                errors.append("FPS is mandatory for image sequences")
    else:
        if not groups.get('fps'):
            errors.append("FPS is mandatory for video files")
    if not re.match(r'^[vV]\d{1,4}$', "v" + groups['version']):
        errors.append("Version token must be 'v' followed by 1-4 digits")
    if groups.get('pixelMapping') is None:
        warnings.append("Optional field 'pixelMapping' is missing")
    return errors, warnings

# --------------------
# New Naming Convention Parser (updated frame_padding to allow 3 or 4 digits, preceded by _ or .)
# --------------------
new_pattern = re.compile(
    r'^(?P<sequence>[A-Za-z]{3,4})(?P<shotNumber>\d{4})_' +
    r'(?P<description>[\w-]+)_' +
    r'(?:(?P<pixelMapping>LL180|LL360))?(?P<resolution>\d+[kK])_' +
    r'(?P<colorspaceGamma>[^_]+)' +
    r'(?:_(?P<fps>\d+))?_' +
    r'v(?P<version>\d{1,4})' +
    r'(?:(?:[_\.](?P<frame_padding>\d{3,4}))?\.(?P<extension>[^.]+))$',
    re.IGNORECASE
)

# --------------------
# Custom QListWidget for Drag and Drop of Folders
# --------------------
class FolderDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
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
# FileScanner Thread (Multithreaded)
# --------------------
class FileScanner(QThread):
    progress = pyqtSignal(int)
    update_preview = pyqtSignal(dict)
    log_message = pyqtSignal(str)

    def __init__(self, directories, delivery_package, file_types):
        super().__init__()
        self.directories = directories
        self.delivery_package = delivery_package
        self.file_types = file_types
        self.data = []
        self.paused = False

    def process_file(self, f):
        basename = os.path.basename(f)
        errors, warnings = validate_filename(basename)
        key = None
        m = new_pattern.match(basename)
        if m:
            groups = m.groupdict()
            key = (groups['sequence'], groups['shotNumber'], groups['description'],
                   groups.get('pixelMapping') or "", groups['resolution'],
                   groups['colorspaceGamma'], groups.get('fps') or "", groups['version'])
        else:
            key = basename
        if errors:
            return {'valid': False, 'errors': errors, 'key': key, 'basename': basename}
        return {'valid': True, 'warnings': warnings, 'data_dict': self.build_data_dict(f, m), 'key': key, 'basename': basename}

    def build_data_dict(self, f, m):
        groups = m.groupdict()
        basename = os.path.basename(f)
        # For image sequences, remove the trailing separator, frame_padding, and extension.
        if groups.get("frame_padding"):
            # Use regex to remove the trailing _ or . plus frame padding and extension.
            version_name = re.sub(r'[_\.]\d{3,4}\.[^.]+$', '', basename)
        else:
            version_name = os.path.splitext(basename)[0]
        shot_name_field = f"{groups['sequence']}_{groups['shotNumber']}"
        try:
            version_number = "v" + format(int(groups['version']), "03d")
        except Exception:
            version_number = "v" + groups['version']
        submitted_for = ""
        delivery_notes = ""
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
        date = datetime.now().strftime("%m/%d/%Y")
        upload_status_field = "Uploaded to Aspera"
        vendor_name_field = "CG Fluids"
        data_dict = {
            "Version Name": version_name,
            "Shot Name": shot_name_field,
            "Version Number": version_number,
            "Submitted For": submitted_for,
            "Delivery Notes": delivery_notes,
            "File Type": file_type_field,
            "Resolution": resolution_field,
            "Duration": duration_field,
            "Delivery Date": date,
            "Delivery Package Name": self.delivery_package,
            "Upload Status": upload_status_field,
            "Vendor Name": vendor_name_field
        }
        return data_dict

    def run(self):
        candidate_files = []
        for directory in self.directories:
            for ft in self.file_types:
                candidate_files.extend(glob.glob(os.path.join(directory, '**', ft), recursive=True))
        candidate_files = sorted(candidate_files)
        total_candidates = len(candidate_files)
        self.log_message.emit(f"Found {total_candidates} candidate files.")
        valid_files = []
        logged_error_keys = set()
        logged_warning_keys = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.process_file, f): f for f in candidate_files}
            processed_count = 0
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                processed_count += 1
                self.progress.emit(int((processed_count / total_candidates) * 100))
                if not result['valid']:
                    if result['key'] not in logged_error_keys:
                        self.log_message.emit(f"File '{result['basename']}' rejected: " + "; ".join(result['errors']))
                        logged_error_keys.add(result['key'])
                    continue
                if result.get('warnings'):
                    if result['key'] not in logged_warning_keys:
                        self.log_message.emit(f"File '{result['basename']}' warning: " + "; ".join(result['warnings']))
                        logged_warning_keys.add(result['key'])
                valid_files.append(result['data_dict'])
        self.log_message.emit(f"{len(valid_files)} files match the naming convention out of {total_candidates} candidates.")
        total_files = len(valid_files)
        processed_files = 0
        headers = ["Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes", 
                   "File Type", "Resolution", "Duration", "Delivery Date", 
                   "Delivery Package Name", "Upload Status", "Vendor Name"]
        self.update_preview.emit({'action': 'init', 'headers': headers})
        sequences = {}  # Key = (Version Name, file type)
        for data_dict in valid_files:
            key_seq = (data_dict["Version Name"], data_dict["File Type"].lower())
            if key_seq in sequences:
                continue
            sequences[key_seq] = True
            row_data = [
                data_dict["Version Name"],
                data_dict["Shot Name"],
                data_dict["Version Number"],
                data_dict["Submitted For"],
                data_dict["Delivery Notes"],
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
            self.progress.emit(int((processed_files / total_files) * 100))
            self.update_preview.emit({'action': 'update', 'row_data': row_data})
        self.progress.emit(100)
        self.update_preview.emit({'action': 'complete'})
        self.log_message.emit("Scanning complete.")

# --------------------
# Main Window
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 975, 800)
        self.setWindowTitle("File Scanner - New CSV Format")
        self.scanner = None
        self.table = None
        self.imageTypeCheckboxes = {}
        self.videoTypeCheckboxes = {}

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Drag and drop the folders you want to scan into the area below.")

        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMinimumHeight(150)

        folder_drop_group = QGroupBox("Drag and Drop Folders (they will be checked by default)")
        self.folder_drop_list = FolderDropListWidget()
        self.folder_drop_list.setMinimumHeight(150)
        drop_layout = QVBoxLayout()
        drop_layout.addWidget(self.folder_drop_list)
        folder_drop_group.setLayout(drop_layout)

        self.remove_button = QPushButton("Remove Selected Folders")
        self.remove_button.setMinimumSize(150, 40)
        self.remove_button.clicked.connect(self.remove_selected_folders)

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

        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        delivery_layout = QHBoxLayout()
        delivery_layout.addWidget(QLabel("Delivery Package Name:"))
        delivery_layout.addWidget(self.delivery_package)

        spacing_before_preview = 20

        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes",
            "File Type", "Resolution", "Duration", "Delivery Date",
            "Delivery Package Name", "Upload Status", "Vendor Name"
        ])

        spacing_after_preview = 20

        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.setMinimumSize(150, 80)
        self.start_scan_button.clicked.connect(self.start_scan)
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.setMinimumSize(150, 80)
        self.save_csv_button.clicked.connect(self.save_csv)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_scan_button)
        button_layout.addWidget(self.save_csv_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

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
        main_layout.addWidget(self.log_window)
        self.setCentralWidget(main_widget)

    def remove_selected_folders(self):
        for i in reversed(range(self.folder_drop_list.count())):
            item = self.folder_drop_list.item(i)
            if item.checkState() == Qt.Checked:
                self.folder_drop_list.takeItem(i)

    def get_checked_drop_folders(self):
        checked = []
        for i in range(self.folder_drop_list.count()):
            item = self.folder_drop_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def append_log(self, msg):
        self.log_window.appendPlainText(msg)

    def start_scan(self):
        selected_dirs = self.get_checked_drop_folders()
        if not selected_dirs:
            QMessageBox.warning(self, "No Folders Selected", "Please check at least one folder to scan.")
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
        self.scanner.log_message.connect(self.append_log)
        self.scanner.start()

    def update_delivery_field(self):
        new_text = self.delivery_package.text().strip()
        if self.scanner:
            self.scanner.delivery_package = new_text
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 9, QTableWidgetItem(new_text))

    def update_preview(self, data):
        if data['action'] == 'init':
            pass
        elif data['action'] == 'update':
            row_data = data['row_data']
            row_count = self.table.rowCount()
            self.table.insertRow(row_count)
            headers = [
                "Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes",
                "File Type", "Resolution", "Duration", "Delivery Date",
                "Delivery Package Name", "Upload Status", "Vendor Name"
            ]
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(row_count, col, item)
        elif data['action'] == 'complete':
            self.status_bar.showMessage("Scan complete.")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def save_csv(self):
        if not self.scanner or not self.scanner.data:
            QMessageBox.information(self, "No Data", "There is no data to save. Please perform a scan first.")
            return
        updated_data = []
        headers = [
            "Version Name", "Shot Name", "Version Number", "Submitted For", "Delivery Notes",
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
