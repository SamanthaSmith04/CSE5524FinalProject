import torch
import tlc
from pathlib import Path
import pandas as pd
from IPython.display import display

# Set up file paths
WORK_DIR = Path(".")  # Current directory
DATASET_YAML = WORK_DIR / "dataset.yaml"

# Verify paths exist
print("Verifying dataset structure...")
print("=" * 50)

if not DATASET_YAML.exists():
    print(f"Could not find {DATASET_YAML}")
    print(f"Current directory: {Path.cwd()}")
    print("Please make sure dataset.yaml is in the current directory")
    raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

print(f"✅ Dataset config: {DATASET_YAML}")
print(f"✅ Working directory: {WORK_DIR.resolve()}")

# Display dataset configuration
print("\n Dataset Configuration:")
print("-" * 50)
with open(DATASET_YAML, "r") as f:
    config_content = f.read()
    print(config_content)

# Count dataset files
train_images = list((WORK_DIR / "train" / "images").glob("*.jpg"))
train_labels = list((WORK_DIR / "train" / "labels").glob("*.txt"))
val_images = list((WORK_DIR / "val" / "images").glob("*.jpg"))
val_labels = list((WORK_DIR / "val" / "labels").glob("*.txt"))
test_images = list((WORK_DIR / "test" / "images").glob("*.jpg"))

print("\n Dataset Statistics:")
print("-" * 50)
print(f"✅ Training:   {len(train_images)} images, {len(train_labels)} labels")
print(f"✅ Validation: {len(val_images)} images, {len(val_labels)} labels")
print(f"✅ Test: {len(test_images)} images")