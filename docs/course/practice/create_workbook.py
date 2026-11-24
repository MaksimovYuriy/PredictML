"""Create the workbook from the CSV printed in module 2; run in project root."""
from pathlib import Path
import csv
from openpyxl import Workbook

path = Path('predictml/data/examples/tabular')
book = Workbook()
about = book.active
about.title = 'about'
about.append(['notice'])
about.append(['SYNTHETIC educational data; not real property prices'])
training = book.create_sheet('training')
with (path/'housing.csv').open(encoding='utf-8', newline='') as stream:
    reader = csv.reader(stream)
    header = next(reader)
    training.append(header)
    for row in reader:
        values = []
        for name, raw in zip(header, row):
            value = None if raw == '' else raw
            if value is not None and name in ('id', 'area', 'price'):
                value = float(value)
                if value.is_integer():
                    value = int(value)
            values.append(value)
        training.append(values)
book.save(path/'housing.xlsx')
print(path/'housing.xlsx')
