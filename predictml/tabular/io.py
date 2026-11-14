from pathlib import Path
import csv
import pandas as pd


def sheets(path):
    path = Path(path)
    if path.suffix.lower() != '.xlsx':
        raise ValueError('sheet selection is supported for XLSX files')
    with pd.ExcelFile(path, engine='openpyxl') as book:
        return book.sheet_names


def read_table(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        if sheet is not None:
            raise ValueError('CSV does not support sheet selection')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            header = next(csv.reader(stream), [])
        frame = pd.read_csv(path, dtype=object, encoding='utf-8-sig')
    elif path.suffix.lower() == '.xlsx':
        names = sheets(path)
        selected = names[0] if sheet is None else sheet
        if selected not in names:
            raise ValueError(f'unknown sheet {selected!r}; available: {names}')
        raw = pd.read_excel(path, sheet_name=selected, header=None, dtype=object, engine='openpyxl')
        if raw.empty:
            raise ValueError('table is empty')
        header = raw.iloc[0].tolist()
        frame = raw.iloc[1:].reset_index(drop=True)
        frame.columns = header
    else:
        raise ValueError('expected CSV or XLSX file')
    if not header or any(not isinstance(c, str) or not c.strip() for c in header):
        raise ValueError('column names must be nonempty strings')
    if len(header) != len(set(header)):
        raise ValueError('duplicate column names')
    if frame.empty:
        raise ValueError('table has no data rows')
    return frame.reset_index(drop=True)


def preview(path, sheet=None, rows=10):
    if type(rows) is not int or rows <= 0:
        raise ValueError('preview rows must be positive')
    frame = read_table(path, sheet)
    return dict(rows=len(frame), columns=list(frame.columns),
                missing=frame.isna().sum().to_dict(), preview=frame.head(rows))
