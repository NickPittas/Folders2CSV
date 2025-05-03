# Makefile for CSV_CGF-Kent_Creator
VERSION=0.21
APP_NAME=CGF_CSV_Creator

# Detect platform
UNAME := $(shell uname)

# Default target - shows help
help:
	@echo "CSV_CGF-Kent_Creator v$(VERSION) Build System"
	@echo "----------------------------------------"
	@echo "Usage:"
	@echo "  make mac     - Build macOS application (only works on macOS)"
	@echo "  make windows - Build Windows executable (only works on Windows)"
	@echo "  make clean   - Remove build artifacts"
	@echo ""
	@echo "Platform requirements:"
	@echo "  - macOS builds must be run on macOS"
	@echo "  - Windows builds must be run on Windows"
	@echo "  - Current platform: $(UNAME)"

# Build for macOS
mac:
ifneq ($(UNAME), Darwin)
	@echo "Error: macOS builds can only be created on macOS"
	@echo "Current platform: $(UNAME)"
	@exit 1
else
	@echo "Building $(APP_NAME) v$(VERSION) for macOS..."
	python3 -m venv venv
	./venv/bin/pip install PyQt5 opencv-python pyinstaller
	@echo "CV2 path: $$(./venv/bin/python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')"
	./venv/bin/pyinstaller --name "$(APP_NAME)" --windowed --clean --add-data="$$(./venv/bin/python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')":cv2 CSV_CGF-Kent_Creator.py
	@echo "✅ macOS build complete! App is in the dist/ folder"
endif

# Build for Windows (only works on Windows)
windows:
ifneq ($(OS), Windows_NT)
	@echo "Error: Windows builds can only be created on Windows"
	@echo "Current platform: $(UNAME)"
	@exit 1
else
	@echo "Building $(APP_NAME) v$(VERSION) for Windows..."
	python -m venv venv
	.\venv\Scripts\pip install PyQt5 opencv-python pyinstaller
	.\venv\Scripts\pyinstaller --name "$(APP_NAME)" --windowed --clean --add-data="$$(.\venv\Scripts\python -c "import cv2, os, sys; sys.stdout.write(os.path.dirname(cv2.__file__))")\;cv2" CSV_CGF-Kent_Creator.py
	@echo "✅ Windows build complete! Executable is in the dist\ folder"
endif

# Clean up build artifacts
clean:
	rm -rf build dist venv
	rm -f *.spec