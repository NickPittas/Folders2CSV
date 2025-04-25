#!/usr/bin/env python2
import os
import csv
from datetime import datetime
import sys
import re
import cv2  # For video duration
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem,
    QVBoxLayout, QHBoxLayout, QProgressBar, QWidget,
    QLineEdit, QGroupBox, QCheckBox, QMessageBox,
    QListWidget, QListWidgetItem, QFileDialog,
    QPlainTextEdit, QGridLayout, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --------------------
# Dark Theme Style Sheet
# --------------------
dark_stylesheet = """
QWidget { background-color: #2d2d30; color: #ffffff; font-family: Arial; }
QPushButton { background-color: #3e3e42; border: 1px solid #565656; padding: 5px; }
QPushButton:hover { background-color: #46464b; }
QLineEdit, QTableWidget, QListWidget, QPlainTextEdit {
    background-color: #3e3e42; color: #ffffff; border: 1px solid #565656;
}
QGroupBox { border: 1px solid #565656; margin-top: 10px; }
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px;
}
QCheckBox { spacing: 5px; }
QProgressBar { border: 1px solid #565656; text-align: center; background-color: #3e3e42; }
QProgressBar::chunk { background-color: #007acc; }
QHeaderView::section { background-color: #3e3e42; padding: 4px; border: 1px solid #565656; }
"""

SEQUENCE_THRESHOLD = 5

def build_pattern(seq_min, seq_max, use_shot, use_pixel, use_fps, ver_min, ver_max):
    parts = []
    parts.append(r'^(?P<sequence>[A-Za-z]{%d,%d})' % (seq_min, seq_max))
    if use_shot:
        parts.append(r'(?:(?P<shotNumber>\d{4})_)?')
    parts.append(r'(?P<description>[\w-]+)_')
    if use_pixel:
        parts.append(r'(?:(?P<pixelMapping>LL180|LL360))?')
    parts.append(r'(?P<resolution>\d+[kK])_')
    parts.append(r'(?P<colorspaceGamma>[^_]+)')
    if use_fps:
        parts.append(r'(?:_(?P<fps>\d+))?_')
    else:
        parts.append(r'_')
    parts.append(r'v(?P<version>\d{%d,%d})' % (ver_min, ver_max))
    parts.append(r'(?:[_\.](?P<frame_padding>\d+)\.(?P<extension>[^.]+)'
                 r'|\.(?P<extension2>[^.]+))$')
    return re.compile(''.join(parts), re.IGNORECASE)

# --------------------
# FolderDropListWidget: supports drag & drop of folders
# --------------------
class FolderDropListWidget(QListWidget):
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

    def __init__(self, directories, delivery_package, file_types,
                 seq_min, seq_max, use_shot, use_pixel, use_fps, ver_min, ver_max):
        super(FileScanner, self).__init__()
        self.directories = directories
        self.delivery_package = delivery_package
        self.file_types = file_types
        self.dir_cache = {}
        self.seq_min, self.seq_max = seq_min, seq_max
        self.use_shot, self.use_pixel, self.use_fps = use_shot, use_pixel, use_fps
        self.ver_min, self.ver_max = ver_min, ver_max
        self.pattern = build_pattern(seq_min, seq_max, use_shot, use_pixel, use_fps, ver_min, ver_max)

    def get_dir_listing(self, directory):
        if directory not in self.dir_cache:
            try:
                self.dir_cache[directory] = os.listdir(directory)
            except:
                self.dir_cache[directory] = []
        return self.dir_cache[directory]

    def validate_filename(self, name):
        errors = []
        if '.' not in name:
            errors.append("Missing file extension")
            return errors
        m = self.pattern.match(name)
        if not m:
            errors.append("Filename does not match expected pattern")
            return errors
        g = m.groupdict()
        if self.use_shot and g.get('shotNumber') and not re.match(r'^\d{4}$', g['shotNumber']):
            errors.append("Shot number must be exactly 4 digits if present")
        if not g['description']:
            errors.append("Description missing")
        if not re.match(r'^\d+[kK]$', g['resolution']):
            errors.append("Resolution must be digits+k")
        if not g['colorspaceGamma']:
            errors.append("ColorspaceGamma missing")
        ext = (g.get('extension') or g.get('extension2')).lower()
        fps = g.get('fps')
        img_exts = ['exr','jpg','tiff','tif','png','tga','psd']
        if self.use_fps:
            if ext in img_exts and g.get('frame_padding') and not fps:
                errors.append("FPS mandatory for image sequences")
            if ext not in img_exts and not fps:
                errors.append("FPS mandatory for video files")
        return errors

    def process_file(self, path):
        name = os.path.basename(path)
        errors = self.validate_filename(name)
        if errors:
            return {'valid': False, 'basename': name, 'errors': errors}
        g = self.pattern.match(name).groupdict()
        ext = (g.get('extension') or g.get('extension2')).lower()
        frame = int(g.get('frame_padding')) if g.get('frame_padding') else None
        common = re.sub(r'([_.]\d+)$','', os.path.splitext(name)[0]) if frame is not None else os.path.splitext(name)[0]
        return {'valid': True, 'data': {
            'path': path,
            'basename': name,
            'directory': os.path.dirname(path),
            'seq': g['sequence'],
            'shot': g.get('shotNumber') or '',
            'ext': ext,
            'res': g['resolution'],
            'frame': frame,
            'common_base': common,
            'version': g['version'],
            'fps': g.get('fps')
        }}

    def run(self):
        def walker(dirs):
            for d in dirs:
                try:
                    for e in os.scandir(d):
                        if e.is_dir():
                            yield from walker([e.path])
                        else:
                            low = e.name.lower()
                            if any(low.endswith(ft[1:]) for ft in self.file_types):
                                yield e.path
                except:
                    continue

        files = list(walker(self.directories))
        total = len(files)
        self.log_message.emit(f"Found {total} candidate files.")

        parsed, done = [], 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(self.process_file, f) for f in files]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                done += 1
                self.progress.emit(int(done/total*100))
                if not res['valid']:
                    self.log_message.emit(f"File '{res['basename']}' rejected: {', '.join(res['errors'])}")
                else:
                    parsed.append(res['data'])

        groups = {}
        for d in parsed:
            key = (d['directory'], d['common_base'], d['ext'])
            groups.setdefault(key, []).append(d)

        headers = [
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ]
        self.update_preview.emit({'action':'init','headers':headers})

        for (directory, base, ext), items in groups.items():
            seq_items = [i for i in items if i['frame'] is not None]
            single_items = [i for i in items if i['frame'] is None]

            if seq_items:
                listing = self.get_dir_listing(directory)
                pat = re.compile(r'^' + re.escape(base) + r'[_.](\d+)\.' + re.escape(ext) + r'$', re.IGNORECASE)
                nums = sorted(int(m.group(1)) for fn in listing if (m := pat.match(fn)))
                runs, cur = [], []
                for n in nums:
                    if not cur or n == cur[-1] + 1:
                        cur.append(n)
                    else:
                        runs.append(cur); cur = [n]
                if cur:
                    runs.append(cur)

                for run in runs:
                    length = run[-1] - run[0] + 1
                    duration = str(length) if length >= SEQUENCE_THRESHOLD else "Still Frame"
                    info = next(i for i in seq_items if i['frame'] == run[0])
                    name = os.path.splitext(info['basename'])[0]
                    shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                    version = 'v' + info['version'].zfill(self.ver_max)

                    row = [
                        name, shot, version, "", "",
                        ext.upper(), info['res'],
                        duration, datetime.now().strftime("%m/%d/%Y"),
                        self.delivery_package,
                        "Uploaded to Aspera", "CG Fluids"
                    ]
                    self.update_preview.emit({'action':'update','row_data':row})

            for info in single_items:
                name = os.path.splitext(info['basename'])[0]
                shot = info['seq'] + ('_' + info['shot'] if info['shot'] else '')
                version = 'v' + info['version'].zfill(self.ver_max)
                duration = "Still Frame"
                if info['ext'] not in ['exr','jpg','tiff','tif','png','tga','psd']:
                    cap = cv2.VideoCapture(info['path'])
                    duration = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    cap.release()

                row = [
                    name, shot, version, "", "",
                    info['ext'].upper(), info['res'],
                    duration, datetime.now().strftime("%m/%d/%Y"),
                    self.delivery_package,
                    "Uploaded to Aspera", "CG Fluids"
                ]
                self.update_preview.emit({'action':'update','row_data':row})

        self.update_preview.emit({'action':'complete'})
        self.log_message.emit("Scan complete.")

# --------------------
# MainWindow UI
# --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(100,100,1000,900)
        self.setWindowTitle("File Scanner - New CSV Format")
        self.scanner = None
        self.imageTypeCheckboxes = {}
        self.videoTypeCheckboxes = {}

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Drag and drop folders below.")

        # Folder drop (fixed)
        folder_group = QGroupBox("Drag & Drop Folders")
        self.folder_list = FolderDropListWidget()
        self.folder_list.setMinimumHeight(80)
        folder_layout = QVBoxLayout()
        folder_layout.addWidget(self.folder_list)
        folder_group.setLayout(folder_layout)

        remove_btn = QPushButton("Remove Selected Folders")
        remove_btn.clicked.connect(self.remove_selected_folders)

        # File-type selectors
        self.select_all = QCheckBox("Select All File Types")
        self.select_img = QCheckBox("Select All Image Files")
        self.select_vid = QCheckBox("Select All Video Files")
        self.select_all.toggled.connect(self.on_select_all)
        self.select_img.toggled.connect(self.on_select_image)
        self.select_vid.toggled.connect(self.on_select_video)
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(self.select_all)
        sel_layout.addWidget(self.select_img)
        sel_layout.addWidget(self.select_vid)

        img_box = QGroupBox("Image File Types")
        img_layout = QHBoxLayout()
        for ext in ['exr','jpg','tiff','tif','png','tga','psd']:
            cb = QCheckBox(ext.upper()); cb.setChecked(True)
            img_layout.addWidget(cb)
            self.imageTypeCheckboxes[ext] = cb
        img_box.setLayout(img_layout)

        vid_box = QGroupBox("Video File Types")
        vid_layout = QHBoxLayout()
        for ext in ['mov','mxf','mp4']:
            cb = QCheckBox(ext.upper()); cb.setChecked(True)
            vid_layout.addWidget(cb)
            self.videoTypeCheckboxes[ext] = cb
        vid_box.setLayout(vid_layout)

        type_group = QGroupBox("File Type Selection")
        type_lay = QHBoxLayout()
        type_lay.addWidget(img_box)
        type_lay.addWidget(vid_box)
        type_group.setLayout(type_lay)

        # Pattern settings
        pat_group = QGroupBox("Filename Pattern Settings")
        pat_layout = QGridLayout()
        pat_layout.addWidget(QLabel("Sequence Length (min – max):"), 0, 0)
        self.seqMin = QSpinBox(); self.seqMin.setRange(1,10); self.seqMin.setValue(3)
        pat_layout.addWidget(self.seqMin, 0, 1)
        self.seqMax = QSpinBox(); self.seqMax.setRange(1,10); self.seqMax.setValue(4)
        pat_layout.addWidget(self.seqMax, 0, 2)
        self.shotCheck = QCheckBox("Enable 4-digit shot number"); self.shotCheck.setChecked(True)
        pat_layout.addWidget(self.shotCheck, 1, 0, 1, 3)
        self.pixCheck = QCheckBox("Enable LL180/360 pixel mapping"); self.pixCheck.setChecked(True)
        pat_layout.addWidget(self.pixCheck, 2, 0, 1, 3)
        self.fpsCheck = QCheckBox("Enable FPS component"); self.fpsCheck.setChecked(True)
        pat_layout.addWidget(self.fpsCheck, 3, 0, 1, 3)
        pat_layout.addWidget(QLabel("Version Digits (min – max):"), 4, 0)
        self.verMin = QSpinBox(); self.verMin.setRange(1,10); self.verMin.setValue(3)
        pat_layout.addWidget(self.verMin, 4, 1)
        self.verMax = QSpinBox(); self.verMax.setRange(1,10); self.verMax.setValue(3)
        pat_layout.addWidget(self.verMax, 4, 2)
        pat_group.setLayout(pat_layout)

        # Delivery package
        del_layout = QHBoxLayout()
        del_layout.addWidget(QLabel("Delivery Package Name:"))
        self.delEdit = QLineEdit()
        self.delEdit.textChanged.connect(self.update_delivery_field)
        del_layout.addWidget(self.delEdit)

        # Preview table
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Version Name","Shot Name","Version Number","Submitted For","Delivery Notes",
            "File Type","Resolution","Duration","Delivery Date",
            "Delivery Package Name","Upload Status","Vendor Name"
        ])
        self.table.setSortingEnabled(True)

        # Progress & buttons
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximum(100)
        self.start_btn = QPushButton("Start Scan"); self.start_btn.clicked.connect(self.start_scan)
        self.save_btn  = QPushButton("Save CSV");  self.save_btn.clicked.connect(self.save_csv)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn); btn_layout.addWidget(self.save_btn)

        # Log window
        self.log_window = QPlainTextEdit(); self.log_window.setReadOnly(True); self.log_window.setMinimumHeight(80)

        # Layout assembly
        main = QVBoxLayout()
        main.addWidget(folder_group)
        main.addWidget(remove_btn)
        main.addLayout(sel_layout)
        main.addWidget(type_group)
        main.addWidget(pat_group)
        main.addLayout(del_layout)
        main.addWidget(self.table)
        main.addWidget(self.progress_bar)
        main.addLayout(btn_layout)
        main.addWidget(self.log_window)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)
        self.setStyleSheet(dark_stylesheet)

    def remove_selected_folders(self):
        for i in reversed(range(self.folder_list.count())):
            it = self.folder_list.item(i)
            if it.checkState() == Qt.Checked:
                self.folder_list.takeItem(i)

    def on_select_all(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            self.select_img.blockSignals(True); self.select_vid.blockSignals(True)
            self.select_img.setChecked(False); self.select_vid.setChecked(False)
            self.select_img.blockSignals(False); self.select_vid.blockSignals(False)

    def on_select_image(self, state):
        if state:
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all.blockSignals(True); self.select_vid.blockSignals(True)
            self.select_all.setChecked(False); self.select_vid.setChecked(False)
            self.select_all.blockSignals(False); self.select_vid.blockSignals(False)

    def on_select_video(self, state):
        if state:
            for cb in self.videoTypeCheckboxes.values(): cb.setChecked(True)
            for cb in self.imageTypeCheckboxes.values(): cb.setChecked(False)
            self.select_all.blockSignals(True); self.select_img.blockSignals(True)
            self.select_all.setChecked(False); self.select_img.setChecked(False)
            self.select_all.blockSignals(False); self.select_img.blockSignals(False)

    def update_delivery_field(self):
        txt = self.delEdit.text().strip()
        if hasattr(self, 'scanner'):
            self.scanner.delivery_package = txt
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 9, QTableWidgetItem(txt))

    def start_scan(self):
        dirs = [self.folder_list.item(i).text()
                for i in range(self.folder_list.count())
                if self.folder_list.item(i).checkState() == Qt.Checked]
        if not dirs:
            QMessageBox.warning(self, "No Folders Selected", "Please select at least one folder.")
            return

        types = []
        for ext, cb in self.imageTypeCheckboxes.items():
            if cb.isChecked(): types.append(f"*.{ext}")
        for ext, cb in self.videoTypeCheckboxes.items():
            if cb.isChecked(): types.append(f"*.{ext}")
        if not types:
            QMessageBox.warning(self, "No File Types", "Please select at least one file type.")
            return

        self.table.setRowCount(0)
        self.progress_bar.setValue(0)

        self.scanner = FileScanner(
            directories=dirs,
            delivery_package=self.delEdit.text().strip(),
            file_types=types,
            seq_min=self.seqMin.value(),
            seq_max=self.seqMax.value(),
            use_shot=self.shotCheck.isChecked(),
            use_pixel=self.pixCheck.isChecked(),
            use_fps=self.fpsCheck.isChecked(),
            ver_min=self.verMin.value(),
            ver_max=self.verMax.value()
        )
        self.scanner.progress.connect(self.progress_bar.setValue)
        self.scanner.update_preview.connect(self.update_preview)
        self.scanner.log_message.connect(self.log_window.appendPlainText)
        self.scanner.start()

    def update_preview(self, data):
        if data['action'] == 'update':
            row = self.table.rowCount()
            self.table.insertRow(row)
            for c, v in enumerate(data['row_data']):
                itm = QTableWidgetItem(str(v))
                itm.setFlags(itm.flags() | Qt.ItemIsEditable)
                self.table.setItem(row, c, itm)
        elif data['action'] == 'complete':
            self.status_bar.showMessage("Scan complete.")

    def save_csv(self):
        if not hasattr(self, 'scanner'):
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
                itm = self.table.item(r, c)
                row.append(itm.text() if itm else "")
            rows.append(row)

        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "*.csv")
        if fn:
            with open(fn, 'w') as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            self.status_bar.showMessage("CSV saved successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
