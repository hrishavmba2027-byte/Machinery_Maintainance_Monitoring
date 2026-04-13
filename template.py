import os
from pathlib import Path

def build_project_structure():
    """Creates the miom_dashboard directory structure and empty files."""
    
    # List of all directories to create
    directories = [
        ".streamlit",
        "core",
        "monitoring",
        "dashboard",
        "data/raw",
        "data/processed",
        "data/live",
        "data/snapshots",
        "checkpoints",
        "logs"
    ]

    # List of all files to create
    files = [
        "app.py",
        "config.py",
        "requirements.txt",
        ".streamlit/config.toml",
        "core/model.py",
        "core/dataset.py",
        "core/preprocessing.py",
        "core/inference.py",
        "monitoring/watcher.py",
        "monitoring/change_detector.py",
        "monitoring/alerting.py",
        "dashboard/fleet_overview.py",
        "dashboard/rul_chart.py",
        "dashboard/triage_table.py",
        "dashboard/engine_detail.py",
        "dashboard/alerts.py",
        "data/raw/train_FD001.txt",
        "data/raw/test_FD001.txt",
        "data/raw/RUL_FD001.txt",
        "checkpoints/best_transformer_FD001.pt",
        "checkpoints/scaler_FD001.pkl",
        "logs/monitoring.log"
    ]

    print("🚀 Building project structure...")

    # 1. Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}/")

    # 2. Create empty files
    for file_path in files:
        path_obj = Path(file_path)
        # Touch creates an empty file. exist_ok=True prevents errors if it already exists
        path_obj.touch(exist_ok=True)
        print(f"📄 Created file: {file_path}")

    print("\n✅ Structure built successfully!")

if __name__ == "__main__":
    build_project_structure()