#!/usr/bin/env python2
import os
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

# Sequence threshold: minimum consecutive frames to count as a sequence
SEQUENCE_THRESHOLD = 5

# --------------------
# Regex pattern: shotNumber optional, pixelMapping optional, tail alternatives
# --------------------
new_pattern = re.compile(r"""
^(?P<sequence>[A-Za-z]{3,4})           # 3-4 letter sequence
(?:(?P<shotNumber>\d{4})_)?            # optional shot number + underscore
(?P<description>[\w-]+)_               # description
(?:(?P<pixelMapping>LL180|LL360))?     # optional pixelMapping
(?P<resolution>\d+[kK])_               # resolution
(?P<colorspaceGamma>[^_]+)             # colorspace+gamma
(?:_(?P<fps>\d+))?_                    # optional fps
v(?P<version>\d{1,4})                  # version
(?:                                    # either frame padding + extension...
   [_\.](?P<frame_padding>\d+)\.(?P<extension>[^.]+)
  |                                    # ...or single-file extension
   \.(?P<extension2>[^.]+)
)$
""", re.IGNORECASE | re.VERBOSE)

# --------------------
# Validate filename
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
    g = m.groupdict()
    # shotNumber check
    if g.get('shotNumber') and not re.match(r'^\d{4}$', g['shotNumber']):
        errors.append("Shot number must be exactly 4 digits if present")
    if not g['description']:
        errors.append("Description missing")
    if not re.match(r'^\d+[kK]$', g['resolution']):
        errors.append("Resolution must be digits+k")
    if not g['colorspaceGamma']:
        errors.append("ColorspaceGamma missing")
    ext = (g.get('extension') or g.get('extension2') or '').lower()
    img_exts = ['exr','jpg','tiff','tif','png','tga','psd']
    if ext in img_exts and g.get('frame_padding') and not g.get('fps'):
        errors.append("FPS mandatory for image sequences")
    if ext not in img_exts and not g.get('fps'):
        errors.append("FPS mandatory for video files")
    if not re.match(r'^[vV]\d{1,4}$', 'v'+g['version']):
        errors.append("Version must be v followed by 1-4 digits")
    return errors, warnings

# --------------------
# Folder drop widget
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
                p = url.toLocalFile()
                if os.path.isdir(p) and p not in [self.item(i).text() for i in range(self.count())]:
                    it = QListWidgetItem(p)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked)
                    self.addItem(it)
            e.acceptProposedAction()
        else:
            e.ignore()

# --------------------
# FileScanner thread
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
        self.dir_cache = {}  # directory listing cache

    def get_dir_listing(self, directory):
        if directory not in self.dir_cache:
            try:
                self.dir_cache[directory] = os.listdir(directory)
            except:
                self.dir_cache[directory] = []
        return self.dir_cache[directory]

    def process_file(self, path):
        name = os.path.basename(path)
        errors, warnings = validate_filename(name)
        if errors:
            return {'valid':False, 'basename':name, 'errors':errors}
        m = new_pattern.match(name)
        g = m.groupdict()
        ext = (g.get('extension') or g.get('extension2')).lower()
        frame = int(g['frame_padding']) if g.get('frame_padding') else None
        common_base = None
        if frame is not None:
            common_base = re.sub(r'([_.]\d+)$','',os.path.splitext(name)[0])
        return {'valid':True, 'data':{
            'path': path,
            'basename': name,
            'directory': os.path.dirname(path),
            'seq': g['sequence'],
            'shot': g.get('shotNumber') or '',
            'desc': g['description'],
            'ext': ext,
            'res': g['resolution'],
            'frame': frame,
            'common_base': common_base,
            'version': g['version'],
            'fps': g.get('fps')
        }}

    def run(self):
        # 1. Walk directories via os.scandir
        def walk_dirs(dirs):
            for d in dirs:
                try:
                    for e in os.scandir(d):
                        if e.is_dir():
                            yield from walk_dirs([e.path])
                        else:
                            n = e.name.lower()
                            if any(n.endswith(ft[1:]) for ft in self.file_types):
                                yield e.path
                except:
                    continue

        files = list(walk_dirs(self.directories))
        total = len(files)
        self.log_message.emit("Found {} candidate files.".format(total))

        # 2. Validate & parse
        parsed = []
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.process_file, f) for f in files]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                done += 1
                self.progress.emit(int(done/total*100))
                if not res['valid']:
                    self.log_message.emit("File '{}' rejected: {}".format(
                        res['basename'], "; ".join(res['errors'])
                    ))
                else:
                    parsed.append(res['data'])

        # 3. Group by directory + base + ext
        groups = {}
        for d in parsed:
            key = (d['directory'], d['common_base'], d['ext'])
            groups.setdefault(key, []).append(d)

        # 4. Init preview
        headers = [
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ]
        self.update_preview.emit({'action':'init','headers':headers})

        # 5. Process groups
        for (directory, base, ext), items in groups.items():
            seq_items = [i for i in items if i['frame'] is not None]
            single_items = [i for i in items if i['frame'] is None]

            # Sequence detection
            if seq_items:
                listing = self.get_dir_listing(directory)
                pat = re.compile(r'^' + re.escape(base) + r'[_\.](\d+)\.' + re.escape(ext) + '$', re.IGNORECASE)
                suffixes = sorted(int(m.group(1)) for fn in listing if (m:=pat.match(fn)))
                # build runs
                runs, cur = [], []
                for s in suffixes:
                    if not cur or s == cur[-1]+1:
                        cur.append(s)
                    else:
                        runs.append(cur); cur=[s]
                if cur: runs.append(cur)

                for run in runs:
                    if len(run) >= SEQUENCE_THRESHOLD:
                        first, last = run[0], run[-1]
                        missing = [str(x) for x in range(first, last+1) if x not in run]
                        if missing:
                            self.log_message.emit(f"Sequence {base} missing frames: {','.join(missing)}")
                        duration = str(last-first+1)
                        info = next(i for i in seq_items if i['frame']==first)
                        version = "v"+info['version'].zfill(3)
                        name = os.path.splitext(info['basename'])[0]
                        shot = info['seq'] + ('_'+info['shot'] if info['shot'] else '')
                        row = [
                            name, shot, version, "", "", ext.upper(),
                            info['res'], duration,
                            datetime.now().strftime("%m/%d/%Y"),
                            self.delivery_package, "Uploaded to Aspera", "CG Fluids"
                        ]
                        self.update_preview.emit({'action':'update','row_data':row})
                    else:
                        for s in run:
                            info = next(i for i in seq_items if i['frame']==s)
                            version = "v"+info['version'].zfill(3)
                            name = os.path.splitext(info['basename'])[0]
                            shot = info['seq'] + ('_'+info['shot'] if info['shot'] else '')
                            duration = "Still Frame"
                            if ext not in ['exr','jpg','tiff','tif','png','tga','psd']:
                                cap = cv2.VideoCapture(info['path'])
                                duration = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                                cap.release()
                            row = [
                                name, shot, version, "", "", ext.upper(),
                                info['res'], duration,
                                datetime.now().strftime("%m/%d/%Y"),
                                self.delivery_package, "Uploaded to Aspera", "CG Fluids"
                            ]
                            self.update_preview.emit({'action':'update','row_data':row})

            # Singles
            for info in single_items:
                version = "v"+info['version'].zfill(3)
                name = os.path.splitext(info['basename'])[0]
                shot = info['seq'] + ('_'+info['shot'] if info['shot'] else '')
                duration = "Still Frame"
                if info['ext'] not in ['exr','jpg','tiff','tif','png','tga','psd']:
                    cap = cv2.VideoCapture(info['path'])
                    duration = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    cap.release()
                row = [
                    name, shot, version, "", "", ext.upper(),
                    info['res'], duration,
                    datetime.now().strftime("%m/%d/%Y"),
                    self.delivery_package, "Uploaded to Aspera", "CG Fluids"
                ]
                self.update_preview.emit({'action':'update','row_data':row})

        # 6. Complete
        self.update_preview.emit({'action':'complete'})
        self.log_message.emit("Scan complete.")

# --------------------
# MainWindow UI
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100, 100, 975, 800)
        self.setWindowTitle("File Scanner - New CSV Format")
        self.scanner = None
        self.imageTypeCheckboxes = {}
        self.videoTypeCheckboxes = {}

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Drag and drop the folders you want to scan below.")

        # Folder drop
        folder_drop_group = QGroupBox("Drag and Drop Folders (checked by default)")
        self.folder_drop_list = FolderDropListWidget()
        self.folder_drop_list.setMinimumHeight(80)  # half height
        drop_layout = QVBoxLayout()
        drop_layout.addWidget(self.folder_drop_list)
        folder_drop_group.setLayout(drop_layout)

        # Remove button
        self.remove_button = QPushButton("Remove Selected Folders")
        self.remove_button.setMinimumSize(50, 40)
        self.remove_button.clicked.connect(self.remove_selected_folders)

        # Master selectors
        self.select_all_checkbox   = QCheckBox("Select All File Types")
        self.select_image_checkbox = QCheckBox("Select All Image Files")
        self.select_video_checkbox = QCheckBox("Select All Video Types")
        self.select_all_checkbox.toggled.connect(self.on_select_all)
        self.select_image_checkbox.toggled.connect(self.on_select_image)
        self.select_video_checkbox.toggled.connect(self.on_select_video)
        select_layout = QHBoxLayout()
        select_layout.addWidget(self.select_all_checkbox)
        select_layout.addWidget(self.select_image_checkbox)
        select_layout.addWidget(self.select_video_checkbox)

        # File-type checkboxes
        image_types = ['exr','jpg','tiff','tif','png','tga','psd']
        video_types = ['mov','mxf','mp4']
        image_box = QGroupBox("Image File Types")
        image_layout = QHBoxLayout()
        for ft in image_types:
            cb = QCheckBox(ft.upper()); cb.setChecked(True)
            image_layout.addWidget(cb); self.imageTypeCheckboxes[ft] = cb
        image_box.setLayout(image_layout)

        video_box = QGroupBox("Video File Types")
        video_layout = QHBoxLayout()
        for ft in video_types:
            cb = QCheckBox(ft.upper()); cb.setChecked(True)
            video_layout.addWidget(cb); self.videoTypeCheckboxes[ft] = cb
        video_box.setLayout(video_layout)

        fileTypeSelectionGroup = QGroupBox("File Type Selection")
        ft_layout = QHBoxLayout()
        ft_layout.addWidget(image_box)
        ft_layout.addWidget(video_box)
        fileTypeSelectionGroup.setLayout(ft_layout)

        # Delivery package
        delivery_layout = QHBoxLayout()
        delivery_label = QLabel("Delivery Package Name:")
        self.delivery_package = QLineEdit()
        self.delivery_package.setPlaceholderText("Enter Delivery Package Name")
        self.delivery_package.textChanged.connect(self.update_delivery_field)
        delivery_layout.addWidget(delivery_label)
        delivery_layout.addWidget(self.delivery_package)

        # Table preview
        self.table = QTableWidget()
        self.table.setMinimumHeight(800)  # double height
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ])

        # Progress & buttons
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximum(100)
        self.start_scan_button = QPushButton("Start Scan"); self.start_scan_button.setMinimumSize(50,40)
        self.start_scan_button.clicked.connect(self.start_scan)
        self.save_csv_button  = QPushButton("Save CSV");  self.save_csv_button.setMinimumSize(50,40)
        self.save_csv_button.clicked.connect(self.save_csv)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_scan_button)
        button_layout.addWidget(self.save_csv_button)

        # Log window
        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMinimumHeight(80)

        # Assemble layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(folder_drop_group)
        main_layout.addWidget(self.remove_button)
        main_layout.addLayout(select_layout)
        main_layout.addWidget(fileTypeSelectionGroup)
        main_layout.addLayout(delivery_layout)
        main_layout.addWidget(self.table)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.log_window)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def remove_selected_folders(self):
        for i in reversed(range(self.folder_drop_list.count())):
            it = self.folder_drop_list.item(i)
            if it.checkState() == Qt.Checked:
                self.folder_drop_list.takeItem(i)

    def get_checked_drop_folders(self):
        return [
            self.folder_drop_list.item(i).text()
            for i in range(self.folder_drop_list.count())
            if self.folder_drop_list.item(i).checkState() == Qt.Checked
        ]

    def update_delivery_field(self):
        txt = self.delivery_package.text().strip()
        if hasattr(self, 'scanner'):
            self.scanner.delivery_package = txt
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 9, QTableWidgetItem(txt))

    def start_scan(self):
        dirs = self.get_checked_drop_folders()
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
        self.table.clearContents(); self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        dp = self.delivery_package.text().strip()
        self.scanner = FileScanner(dirs, dp, types)
        self.scanner.progress.connect(self.progress_bar.setValue)
        self.scanner.update_preview.connect(self.update_preview)
        self.scanner.log_message.connect(self.log_window.appendPlainText)
        self.scanner.start()

    def update_preview(self, data):
        if data['action'] == 'update':
            r = self.table.rowCount(); self.table.insertRow(r)
            for c,v in enumerate(data['row_data']):
                itm = QTableWidgetItem(str(v))
                itm.setFlags(itm.flags()|Qt.ItemIsEditable)
                self.table.setItem(r,c,itm)
        elif data['action'] == 'complete':
            self.status_bar.showMessage("Scan complete.")

    def save_csv(self):
        if not hasattr(self,'scanner'):
            QMessageBox.information(self, "No Data", "No scan data to save.")
            return
        headers = [
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ]
        rows = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(len(headers)):
                itm = self.table.item(r,c)
                row.append(itm.text() if itm else "")
            rows.append(row)
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "*.csv")
        if fn:
            with open(fn,'w') as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            self.status_bar.showMessage("CSV saved successfully!")

    def on_select_all(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            self.select_image_checkbox.blockSignals(True)
            self.select_video_checkbox.blockSignals(True)
            self.select_image_checkbox.setChecked(False)
            self.select_video_checkbox.setChecked(False)
            self.select_image_checkbox.blockSignals(False)
            self.select_video_checkbox.blockSignals(False)

    def on_select_image(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all_checkbox.blockSignals(True)
            self.select_video_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_video_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            self.select_video_checkbox.blockSignals(False)

    def on_select_video(self, state):
        if state:
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all_checkbox.blockSignals(True)
            self.select_image_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_image_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            self.select_image_checkbox.blockSignals(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
