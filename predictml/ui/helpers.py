"""Small UI adapters; calculations remain in the public core APIs."""
import hashlib
from pathlib import Path
import uuid


def save_upload(upload, workspace, *, name=None):
    data = upload.getvalue()
    filename = Path(name or upload.name).name
    if filename in ('', '.', '..'):
        raise ValueError('Некорректное имя файла')
    directory = Path(workspace)/'uploads'/hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/filename
    if not path.exists():
        path.write_bytes(data)
    return path


def save_group(files, workspace):
    directory = Path(workspace)/'uploads'/str(uuid.uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    names = set()
    for upload in files:
        name = Path(upload.name).name
        if name in names:
            raise ValueError(f'Повторное имя файла: {name}')
        names.add(name)
        (directory/name).write_bytes(upload.getvalue())
    return directory


def layers(value):
    try:
        result = tuple(int(v.strip()) for v in value.split(',') if v.strip())
    except ValueError as exc:
        raise ValueError('Скрытые слои: укажите целые числа через запятую, например 64,32') from exc
    if not result or any(v <= 0 for v in result):
        raise ValueError('Скрытые слои должны быть положительными')
    return result


def version_label(record):
    return f"{record['task']} · {record['model_kind']} · {record['target']} · {record['created_at'][:19]} UTC · {record['id'][:8]}"
