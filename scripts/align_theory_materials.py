"""Map the author's existing theory DOCX and three PPTX; never edit originals.

python scripts/align_theory_materials.py --book PATH.docx --presentations-dir DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
# References are numbers from the original book, not presentation identifiers.
ROWS={
 '1.1':[
 ('Как работать с пособием',),('Результаты обучения',),('1.1',),('1.2','1.5'),('1.3',),('1.2','1.5'),('1.5','3.8','4.4'),('2.12','3.8'),('1.5','3.9'),('1.6','4.1'),('1.7','2.12','4.2'),('1.7','4.3'),('3.15',),('1.2','1.5','3.7','3.8'),('1.13','4.4','6')],
 '1.2':[
 ('1.4',),('Результаты обучения',),('1.4',),('1.4.1',),('1.8','1.9','1.10','1.11','1.12'),('1.4.2',),('1.4.3',),('1.4.3',),('1.4.4',),('1.4.5',),('1.4.5',),('1.4.6','2.1','2.5','2.13'),('1.4.6',),('1.4.7','1.11','2.13'),('1.13','3.15','3.16')],
 '1.3':[
 ('2','3'),('Результаты обучения',),('2.2',),('2.3',),('2.4',),('2.5','2.10','2.11'),('2.6','2.7','2.11'),('2.9','2.10'),('3.1','3.4','3.6'),('3.2','3.3','3.13'),('3.5','3.6'),('3.7','3.8','3.11'),('3.10','2.7','3.11'),('3.12','3.13','3.14','3.15'),('2.14','3.20','6')]
}
NOTES={
 ('1.1',7):('частично','Горизонт показан отдельной таблицей на слайде; в книге объяснение распределено между постановкой времени и ограничениями. Нет отдельного подраздела с теми же примерами 1/3/12.'),
 ('1.1',9):('покрыто','Тема утечки вынесена в первую презентацию; развёрнутый перечень книги находится позже, в §3.9. Это изменение порядка, а не отсутствие темы.'),
 ('1.1',11):('требует согласования','В книге основной пример описан через booking_ratio, на слайде допускаются количество и demand_ratio; текущие пособия 2/3 используют total_bookings.'),
 ('1.2',5):('частично','Один слайд сворачивает пять подразделов линейной регрессии. Обучение коэффициентов и мини-пример подробнее раскрываются при чтении книги.'),
 ('1.2',11):('частично','В §1.4.5 есть идея ARIMA и стационарности, но компоненты AR/I/MA раскрыты на слайде подробнее, чем в книге.'),
 ('1.2',13):('частично','В §1.4.6 есть обработка последовательности и её порядок; скрытое состояние и механизмы памяти RNN/LSTM на слайде конкретнее.'),
 ('1.3',6):('требует согласования','Пример выхода demand_ratio расходится с booking_ratio в §1.7/§4.2 и с total_bookings основного demand-v2. Числа 7/10 обозначают примеры архитектур, а не универсальную ширину любой таблицы.'),
 ('1.3',7):('частично','Dropout присутствует в схеме, но его train/eval-режим и линейный регрессионный выход подробнее объясняются только в §2.7/§2.11.'),
 ('1.3',9):('частично','Цикл показан обзорно; обнуление градиентов из §3.4 и устройство Adam из §3.6 не раскрываются отдельными пунктами.'),
 ('1.3',12):('частично','Роли частей совпадают. Временной split и подробный выбор best_state/early stopping требуют чтения §3.8/§3.11.'),
 ('1.3',14):('частично','MAE/MSE/wMAPE и baseline совпадают; RMSE присутствует в §3.13, но отсутствует в таблице слайда. R² в этих источниках отдельно не раскрыт.')
}
ISSUES=[
 dict(id='structure',priority='организация',where='Структура модуля; темы 1–3; презентации 1.1–1.3',finding='Нет соответствия «одна тема книги — одна презентация». Первая тема книги разделена между 1.1 и 1.2, а темы MLP и обучения объединены в 1.3.',recommendation='Сохранить полный текст и использовать маршрут чтения по презентациям из карты; при редактуре добавить такие переходы в оглавление, не сокращая книгу до слайдов.'),
 dict(id='demand-target',priority='содержательное согласование',where='§1.2, §1.7, §2.5, §4.2; 1.1 слайд 11; 1.3 слайд 6',finding='Внутри теории используются booking_ratio и demand_ratio, а новые пособия и основная программа прогнозируют total_bookings. Это разные определения цели, а не варианты одного названия.',recommendation='Для единого рабочего примера согласованно описать прямой прогноз количества; относительный пример при необходимости явно пометить историческим. Исправлять одновременно книгу и соответствующие слайды, а не только переименовать одну подпись.'),
 dict(id='module-boundary',priority='содержательное согласование',where='§3.11, абзац о полной реализации обучения; для сравнения §2.10',finding='§3.11 относит полную реализацию обучения и продолжения в третий модуль. Сейчас код реализуется в пособии 2, а модуль 3 запускает готовую программу и исследует результат.',recommendation='Заменить переход: реализация цикла и обновления — модуль 2; настройка, выполнение и оценка — модуль 3.'),
 dict(id='coverage-detail',priority='разная глубина',where='§1.4.5 и §1.4.6; 1.2 слайды 11 и 13',finding='Некоторые конкретные пояснения AR/I/MA и RNN/LSTM подробнее в презентации, чем в основном тексте.',recommendation='Добавить короткие пояснения в соответствующие подразделы книги либо явно обозначить слайды как расширение. Новых алгоритмов в программе это не требует.'),
 dict(id='third-deck',priority='объём занятия',where='Темы 2 и 3 книги; презентация 1.3',finding='Одна презентация объединяет архитектуру и обучение с оценкой. Отдельных слайдов о числе параметров, Adam, best_state и воспроизводимости нет.',recommendation='Не удалять эти темы из книги. Выделить их для чтения/разбора по карте. Если всё нужно подробно объяснять на экране, обсудить четвёртую презентацию модуля 1; исходные 3×15 сейчас сохранены.'),
 dict(id='metrics',priority='дополнение перехода',where='§3.13–§3.14; 1.3 слайд 14; пособия 2/3',finding='RMSE есть в книге, но не в таблице слайда; R² появляется в новых практических материалах. Для wMAPE книга требует ненулевой знаменатель, а практическая программа показывает null/пустую ячейку при неопределённости.',recommendation='Согласовать краткую интерпретацию RMSE/R² и отображение неопределённой метрики перед переходом к отчётам программы. Это не означает, что каждую метрику надо помещать на один существующий слайд.'),
 dict(id='audience',priority='редактура',where='Титульные сведения, поле «Уровень»',finding='В исходнике указан вводный курс для студентов и начинающих разработчиков; текущая согласованная аудитория — аспиранты со знанием Python.',recommendation='Обновить сведения об аудитории и предпосылках при редактуре; вводный характер объяснения ML сам по себе не является содержательной ошибкой.')
]


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--book',type=Path,default=Path.home()/'Модуль_1_Теоретические_основы_прогнозирования.docx');parser.add_argument('--presentations-dir',type=Path,default=Path.home());args=parser.parse_args()
 sections=[];current=None
 with ZipFile(args.book) as z:
  for i,p in enumerate(ET.fromstring(z.read('word/document.xml')).iter(W+'p'),1):
   text=''.join(n.text or '' for n in p.iter(W+'t'));style=p.find(W+'pPr/'+W+'pStyle');style=style.get(W+'val') if style is not None else ''
   if style.startswith('Heading'):
    if current:sections.append(current)
    match=re.match(r'^(\d+(?:\.\d+)*)\s',text)
    current=dict(id=match[1] if match else text,title=text,paragraph=i,text=[])
   elif current and text:current['text'].append(text)
  if current:sections.append(current)
 by={s['id']:s for s in sections};rows=[];sources=[dict(name=args.book.name,sha256=hashlib.sha256(args.book.read_bytes()).hexdigest(),kind='book')]
 for number in (1,2,3):
  deck=f'1.{number}';found=list(args.presentations_dir.glob(f'Презентация_1_{number}_*.pptx'))
  if len(found)!=1:raise ValueError(f'expected one presentation {deck}, got {found}')
  path=found[0];sources.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind='presentation',presentation=deck))
  with ZipFile(path) as z:
   for slide,refs in enumerate(ROWS[deck],1):
    e=ET.fromstring(z.read(f'ppt/slides/slide{slide}.xml'));texts=[v.text or '' for v in e.iter() if v.tag.endswith('}t')]
    # The source title is the third text item after the repeated header and number.
    title=texts[2] if slide>1 else texts[1]
    status,comment=NOTES.get((deck,slide),('покрыто','Основная тема слайда раскрыта в указанных разделах книги.'))
    if slide in (1,2):status='служебный';comment='Обложка или результаты занятия; это не отдельная содержательная тема.'
    if slide==15:status='обобщение';comment='Итоги нескольких разделов, а не новое содержание.'
    rows.append(dict(module=1,section='; '.join(refs),title=title,presentation=deck,slide=slide,slides=str(slide),test=deck,status=status,comment=comment,
                     book_sections=[dict(id=ref,title=by[ref]['title'],paragraph=by[ref]['paragraph']) for ref in refs]))
 # Reverse map includes every numbered heading, and supplementary reading is explicit.
 reverse=[]
 for s in sections:
  if not re.fullmatch(r'\d+(?:\.\d+)*',s['id']):continue
  refs=[f"{r['presentation']}:{r['slide']}" for r in rows if s['id'] in [b['id'] for b in r['book_sections']]]
  reverse.append(dict(section=s['id'],title=s['title'],paragraph=s['paragraph'],slides=refs,
                      status=('структурный заголовок; содержание раскрыто в подразделах' if s['id'] in ('1','2','3','4') else 'есть прямое соответствие' if refs else 'нет отдельной прямой ссылки; чтение по пособию')))
 result=dict(module=1,sources=sources,conclusion='Основные понятия согласуются; деление книги и презентаций не является взаимно однозначным. Найдены конкретные пункты согласования с практическими модулями.',slides=rows,book_sections=reverse,issues=ISSUES,
             method='Сопоставлено текстовое содержание исходных DOCX/PPTX. Номера абзацев — по word/document.xml, не номера страниц. Авторские оригиналы не изменены; оформление не оценивалось.')
 target=ROOT/'docs/course/source/theory-alignment.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Mapped 45 slides and',len(reverse),'numbered book headings; originals unchanged.')


if __name__=='__main__':main()
