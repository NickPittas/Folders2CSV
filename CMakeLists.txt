cmake_minimum_required(VERSION 3.20)
project(csv_creator NONE)                 # “NONE” – no C/C++ language enabled

# Where the script lives
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/CSV_CGF-Kent_Creator.py)

# Location of the venv that contains PyInstaller (see step 3)
set(VENV ${CMAKE_CURRENT_SOURCE_DIR}/.venv)

# Helper to call PyInstaller reproducibly
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist/CSV_CGF-Kent_Creator.exe
    COMMAND ${VENV}/Scripts/python -m PyInstaller
            --noconfirm --onefile --windowed        # ⬅ GUI app; drop --windowed for console
            --name CSV_CGF-Kent_Creator
            --icon ${CMAKE_CURRENT_SOURCE_DIR}/icon.ico  # optional
            ${SRC}
    DEPENDS ${SRC}
    COMMENT "Bundling CSV_CGF-Kent_Creator.exe with PyInstaller"
)

add_custom_target(package ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dist/CSV_CGF-Kent_Creator.exe)
