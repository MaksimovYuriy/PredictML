"""Copy canonical repository examples into package assets before a release."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
destination=root/'predictml/data/examples'
shutil.copytree(root/'examples',destination,dirs_exist_ok=True)
print(destination)
