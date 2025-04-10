# File Scanner with Drag-and-Drop Folder Selection and CSV Export

This Python script provides a dark-themed graphical user interface (GUI) built with PyQt5. It allows you to:

- **Drag and Drop Folders:** Easily add folders you want to scan by dragging and dropping them into the application. Each dropped folder appears as a checkable item.
- **Remove Folders:** Remove unwanted folders using the "Remove Selected Folders" button.
- **Select File Types:** Choose which file types to scan. File types are grouped into "Image File Types" and "Video File Types," with additional "Select All" options for each group as well as a global "Select All File Types" option.
- **Preview and Edit Results:** View the scanned file details (such as file name, version, type, file size, and more) in an editable preview table. Any manual edits you make will be saved to the CSV file.
- **Export CSV:** Save the scanned (and optionally edited) file information to a CSV file.

## Features

- **Drag-and-Drop Folder Selection:**  
  Drag folders into the dedicated drop area. Folders will be automatically added as checkable items (defaulting to checked). You can later remove any folder by selecting it and clicking the **Remove Selected Folders** button.

- **File Type Selection:**  
  The **File Type Selection** section is bordered and divided into two subsections:
  - **Image File Types:** e.g., EXR, JPG, TIFF, TIF, PNG, TGA, PSD.
  - **Video File Types:** e.g., MOV, MXF, MP4.
  
  There are “Select All” checkboxes for both groups as well as a global option for all file types.

- **Editable Preview Table:**  
  The preview table displays the scan results and is fully editable. Any changes made in the table are saved when exporting to CSV.

- **CSV Export:**  
  After scanning (and optionally editing the preview), export the results to a CSV file using the **Save CSV** button.

- **User-Friendly Layout:**  
  The interface features clear spacing between sections, and the Scan and Save buttons are located at the bottom of the window.

## Requirements

The script requires Python 3.x. The provided `requirements.txt` file includes the following modules:
altgraph==0.17.4 auto-py-to-exe==2.46.0 bottle==0.13.2 bottle-websocket==0.2.9 certifi==2025.1.31 cffi==1.17.1 charset-normalizer==3.4.1 Eel==0.18.1 et_xmlfile==2.0.0 future==1.0.0 gevent==24.11.1 gevent-websocket==0.10.1 greenlet==3.1.1 idna==3.10 numpy==2.1.3 opencv-python==4.10.0.84 openpyxl==3.1.5 packaging==24.2 pefile==2023.2.7 pillow==11.0.0 psutil==6.1.0 pycparser==2.22 pyinstaller==6.12.0 pyinstaller-hooks-contrib==2025.1 pyparsing==3.2.1 PyQt5==5.15.11 PyQt5-Qt5==5.15.2 PyQt5_sip==12.15.0 pywin32-ctypes==0.2.3 requests==2.32.3 setuptools==75.8.0 typing_extensions==4.12.2 urllib3==2.3.0 zope.event==5.0 zope.interface==7.2

> **Note:** The script mainly uses PyQt5. Many of the additional modules are included in the requirements file for packaging or additional functionality (for example, if you decide to bundle the script as an executable). If you only wish to run the script, PyQt5 is the primary dependency.

## Installation

1. **Clone or download the repository.**

2. **Install the required Python modules** by running:

   ```bash
   pip install -r requirements.txt
## Run
   ```bash
   python your_script_name.py
```
## Create Executable
  ```pyinstaller --onefile -w --name "Folder2CSV v9" '.\Folder2CSV v9.py'
```


## How to Use

**Drag and Drop Folders:**
Open the application and drag the folders you wish to scan into the designated drop area. Each folder will appear as a checked item. If you want to remove any folder, select it and click Remove Selected Folders.

**Select File Types:**
In the File Type Selection section, use the provided checkboxes to choose which file types to scan. Use the "Select All" options to quickly check/uncheck the desired groups.

**Enter Delivery Package Name:**
Provide a delivery package name in the corresponding field. This name will appear in the preview table and be included in the CSV output.

**Start Scan:**
Click the Start Scan button at the bottom. The script will scan the selected folders for the chosen file types and display the results in an editable preview table.

**Review and Edit the Preview:**
The preview table shows details like file name, version, file type, file size, etc. You can edit any cell directly.

**Save CSV:**
Once satisfied, click the Save CSV button. The script will update its data from the current table content and save it to a CSV file.
