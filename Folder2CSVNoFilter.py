#!/usr/bin/env python2
import os
import csv
from datetime import datetime
import sys
import re
import cv2  # For video duration
import concurrent.futures
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget, QLineEdit, QGroupBox, QCheckBox,
    QMessageBox, QListWidget, QListWidgetItem, QFileDialog, QPlainTextEdit
)
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
# Widget for dragging/dropping folders
# --------------------
class FolderDropListWidget(QListWidget):
    def __init__(self, parent=None):
        super(FolderDropListWidget, self).__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    existing = [self.item(i).text() for i in range(self.count())]
                    if path not in existing:
                        it = QListWidgetItem(path)
                        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                        it.setCheckState(Qt.Checked)
                        self.addItem(it)
            e.acceptProposedAction()
        else:
            e.ignore()

# --------------------
# Background thread to scan files without filename validation
# --------------------
class FileScanner(QThread):
    progress = pyqtSignal(int)
    update_preview = pyqtSignal(dict)
    log_message = pyqtSignal(str)

    def __init__(self, directories, delivery_package, file_types):
        super(FileScanner, self).__init__()
        self.directories = directories
        self.delivery_package = delivery_package
        self.file_types = file_types

        # regex to extract trailing "_1234" frame suffix
        self.suffix_re = re.compile(r'^(.*?)[_.](\d+)$')
        # regex to extract version token v### anywhere in stem
        self.version_re = re.compile(r'v(\d+)', re.IGNORECASE)

    def run(self):
        # 1) Walk directories via os.scandir
        def walk_dirs(dirs):
            for d in dirs:
                try:
                    for e in os.scandir(d):
                        if e.is_dir():
                            yield from walk_dirs([e.path])
                        else:
                            name_lower = e.name.lower()
                            for ft in self.file_types:
                                if name_lower.endswith(ft[1:].lower()):
                                    yield e.path
                                    break
                except:
                    continue

        all_files = list(walk_dirs(self.directories))
        total = len(all_files)
        self.log_message.emit(f"Found {total} files.")

        # 2) Parse out frame suffix and group
        infos = []
        for idx, path in enumerate(all_files, 1):
            self.progress.emit(int(idx / float(total) * 100))
            basename = os.path.basename(path)
            stem, ext = os.path.splitext(basename)
            ext = ext[1:].upper()
            # frame suffix?
            m = self.suffix_re.match(stem)
            if m:
                base = m.group(1)
                frame = int(m.group(2))
                infos.append({
                    'directory': os.path.dirname(path),
                    'common_base': base,
                    'frame': frame,
                    'basename': basename,
                    'stem': stem,
                    'ext': ext,
                    'path': path
                })
            else:
                infos.append({
                    'directory': os.path.dirname(path),
                    'common_base': None,
                    'frame': None,
                    'basename': basename,
                    'stem': stem,
                    'ext': ext,
                    'path': path
                })

        # 3) Group into sequences and singles
        seq_groups = defaultdict(list)
        singles = []
        for info in infos:
            if info['common_base'] is not None:
                key = (info['directory'], info['common_base'], info['ext'])
                seq_groups[key].append(info)
            else:
                singles.append(info)

        # 4) Initialize preview
        headers = [
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ]
        self.update_preview.emit({'action': 'init', 'headers': headers})

        # 5) Process each sequence group
        for (directory, base, ext), group in seq_groups.items():
            if len(group) > 1:
                # sequence: count frames
                frames = sorted(info['frame'] for info in group)
                duration = str(len(frames))
                info0 = min(group, key=lambda i: i['frame'])
                # extract version
                vm = self.version_re.search(info0['stem'])
                version_number = ("v" + vm.group(1).zfill(3)) if vm else ""
                row = [
                    base, "", version_number, "", "",
                    ext, "", duration,
                    datetime.now().strftime("%m/%d/%Y"),
                    self.delivery_package, "Uploaded to Aspera", "CG Fluids"
                ]
                self.update_preview.emit({'action': 'update', 'row_data': row})
            else:
                singles.append(group[0])

        # 6) Process pure singles
        for info in singles:
            vm = self.version_re.search(info['stem'])
            version_number = ("v" + vm.group(1).zfill(3)) if vm else ""
            duration = "Still Frame"
            if info['ext'] not in ['EXR','JPG','TIFF','TIF','PNG','TGA','PSD']:
                try:
                    cap = cv2.VideoCapture(info['path'])
                    duration = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    cap.release()
                except:
                    duration = ""
            row = [
                info['stem'], "", version_number, "", "",
                info['ext'], "", duration,
                datetime.now().strftime("%m/%d/%Y"),
                self.delivery_package, "Uploaded to Aspera", "CG Fluids"
            ]
            self.update_preview.emit({'action': 'update', 'row_data': row})

        # 7) Complete
        self.update_preview.emit({'action': 'complete'})
        self.log_message.emit("Scan complete.")

# --------------------
# MainWindow UI Setup
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 975, 800)
        self.setWindowTitle("File Scanner - No Filename Checks")
        self.scanner = None

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Drag and drop the folders you want to scan below.")

        # Folder drop
        folder_group = QGroupBox("Drag & Drop Folders (checked by default)")
        self.folder_list = FolderDropListWidget()
        self.folder_list.setMinimumHeight(40)
        fg_layout = QVBoxLayout()
        fg_layout.addWidget(self.folder_list)
        folder_group.setLayout(fg_layout)

        # Remove button
        remove_btn = QPushButton("Remove Selected Folders")
        remove_btn.setMinimumSize(150, 40)
        remove_btn.clicked.connect(self.remove_selected_folders)

        # Master selectors
        self.select_all_cb   = QCheckBox("Select All File Types")
        self.select_image_cb = QCheckBox("Select All Image Files")
        self.select_video_cb = QCheckBox("Select All Video Files")
        self.select_all_cb.toggled.connect(self.on_select_all)
        self.select_image_cb.toggled.connect(self.on_select_image)
        self.select_video_cb.toggled.connect(self.on_select_video)
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(self.select_all_cb)
        sel_layout.addWidget(self.select_image_cb)
        sel_layout.addWidget(self.select_video_cb)

        # File-type checkboxes
        image_exts = ['exr','jpg','tiff','tif','png','tga','psd']
        video_exts = ['mov','mxf','mp4']
        img_box = QGroupBox("Image File Types")
        img_layout = QHBoxLayout()
        self.imageTypeCheckboxes = {}
        for ft in image_exts:
            cb = QCheckBox(ft.upper()); cb.setChecked(True)
            img_layout.addWidget(cb); self.imageTypeCheckboxes[ft] = cb
        img_box.setLayout(img_layout)
        vid_box = QGroupBox("Video File Types")
        vid_layout = QHBoxLayout()
        self.videoTypeCheckboxes = {}
        for ft in video_exts:
            cb = QCheckBox(ft.upper()); cb.setChecked(True)
            vid_layout.addWidget(cb); self.videoTypeCheckboxes[ft] = cb
        vid_box.setLayout(vid_layout)
        types_group = QGroupBox("File Type Selection")
        tt_layout = QHBoxLayout()
        tt_layout.addWidget(img_box); tt_layout.addWidget(vid_box)
        types_group.setLayout(tt_layout)

        # Delivery package input
        dl_layout = QHBoxLayout()
        dl_label = QLabel("Delivery Package Name:")
        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        dl_layout.addWidget(dl_label); dl_layout.addWidget(self.delivery_package)

        # CSV preview table
        self.table = QTableWidget()
        self.table.setMinimumHeight(800)
        headers = [
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Progress & buttons
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        scan_btn = QPushButton("Start Scan"); scan_btn.setMinimumSize(50,40)
        scan_btn.clicked.connect(self.start_scan)
        save_btn = QPushButton("Save CSV"); save_btn.setMinimumSize(50,40)
        save_btn.clicked.connect(self.save_csv)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(scan_btn); btn_layout.addWidget(save_btn)

        # Log window
        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMinimumHeight(80)

        # Assemble layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(folder_group)
        main_layout.addWidget(remove_btn)
        main_layout.addLayout(sel_layout)
        main_layout.addWidget(types_group)
        main_layout.addLayout(dl_layout)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.log_window)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def remove_selected_folders(self):
        for i in reversed(range(self.folder_list.count())):
            it = self.folder_list.item(i)
            if it.checkState() == Qt.Checked:
                self.folder_list.takeItem(i)

    def get_checked_folders(self):
        return [
            self.folder_list.item(i).text()
            for i in range(self.folder_list.count())
            if self.folder_list.item(i).checkState() == Qt.Checked
        ]

    def on_select_all(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            self.select_image_cb.blockSignals(True)
            self.select_video_cb.blockSignals(True)
            self.select_image_cb.setChecked(False)
            self.select_video_cb.setChecked(False)
            self.select_image_cb.blockSignals(False)
            self.select_video_cb.blockSignals(False)

    def on_select_image(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all_cb.blockSignals(True)
            self.select_video_cb.blockSignals(True)
            self.select_all_cb.setChecked(False)
            self.select_video_cb.setChecked(False)
            self.select_all_cb.blockSignals(False)
            self.select_video_cb.blockSignals(False)

    def on_select_video(self, state):
        if state:
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all_cb.blockSignals(True)
            self.select_image_cb.blockSignals(True)
            self.select_all_cb.setChecked(False)
            self.select_image_cb.setChecked(False)
            self.select_all_cb.blockSignals(False)
            self.select_image_cb.blockSignals(False)

    def start_scan(self):
        dirs = self.get_checked_folders()
        if not dirs:
            QMessageBox.warning(self, "No Folders Selected", "Please select at least one folder.")
            return
        types = []
        for ext, cb in self.imageTypeCheckboxes.items():
            if cb.isChecked(): types.append("*.%s" % ext)
        for ext, cb in self.videoTypeCheckboxes.items():
            if cb.isChecked(): types.append("*.%s" % ext)
        if not types:
            QMessageBox.warning(self, "No File Types", "Please select at least one file type.")
            return

        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        dp = self.delivery_package.text().strip()
        self.scanner = FileScanner(dirs, dp, types)
        self.scanner.progress.connect(self.progress_bar.setValue)
        self.scanner.update_preview.connect(self.handle_update_preview)
        self.scanner.log_message.connect(self.log_window.appendPlainText)
        self.scanner.start()

    def handle_update_preview(self, data):
        action = data.get('action')
        if action == 'init':
            self.table.clearContents()
            self.table.setRowCount(0)
        elif action == 'update':
            self._add_row(data['row_data'])
        elif action == 'complete':
            self.status_bar.showMessage("Scan complete.")

    def _add_row(self, row_data):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(row_data):
            itm = QTableWidgetItem(str(v))
            itm.setFlags(itm.flags() | Qt.ItemIsEditable)
            self.table.setItem(r, c, itm)

    def update_delivery_field(self, text):
        # update scanner attribute
        if hasattr(self, 'scanner'):
            self.scanner.delivery_package = text
        # update existing table rows
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 9, QTableWidgetItem(text))

    def save_csv(self):
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        rows = []
        for r in range(self.table.rowCount()):
            rows.append([self.table.item(r, c).text() if self.table.item(r, c) else ""
                         for c in range(len(headers))])
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "*.csv")
        if fn:
            # prevent blank lines by specifying newline=''
            with open(fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            self.status_bar.showMessage("CSV saved successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
