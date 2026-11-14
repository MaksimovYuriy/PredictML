"""Packaged synthetic data, available independently of a repository checkout."""
from pathlib import Path
import shutil


def examples_dir():
    return Path(__file__).with_name('data')/'examples'


def export_examples(destination):
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('demo destination must be empty; existing files are not overwritten')
    shutil.copytree(examples_dir(),destination,dirs_exist_ok=True)
    return destination.resolve()
