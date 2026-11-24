"""Build self-contained DOCX manuals with full application listings and real results.

.venv/bin/python scripts/build_course_manuals.py
.venv/bin/python scripts/build_course_manuals.py --extract BOOK.docx --output NEW_DIR
The extractor reads only the DOCX; it never copies the author's application files.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
COURSE=ROOT/'docs/course'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
XML='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
ORDER={
 '2.1':['__init__.py','config.py','data.py','preprocessing.py'],
 '2.2':['models.py','prediction.py','evaluation.py','training.py'],
 '2.3':['scenarios/__init__.py','scenarios/materials.py','scenarios/serviceflow.py','scenarios/demand.py','scenarios/price.py','storage/__init__.py','storage/bundles.py','storage/repository.py','compatibility/__init__.py','compatibility/serialization.py','compatibility/legacy.py','compatibility/materials.py'],
 '2.4':['updating.py','reporting.py','workflow.py','demo.py','cli.py','__main__.py','ui/__init__.py','ui/app.py','ui/launcher.py']}
SLIDE_MAP={
 '2.1.1':'1–2','2.1.2':'3','2.1.3':'4–5','2.1.4':'6–7','2.1.5':'5–6','2.1.6':'8–13','2.1.7':'14–15',
 '2.2.1':'1–5','2.2.2':'12','2.2.3':'6','2.2.4':'7–10','2.2.5':'3–4, 11','2.2.6':'13–15',
 '2.3.1':'1–2','2.3.2':'3–6','2.3.3':'7–8','2.3.4':'9','2.3.5':'10–12','2.3.6':'13','2.3.7':'14–15',
 '2.4.1':'1–2','2.4.2':'3–5','2.4.3':'6–8','2.4.4':'9–11','2.4.5':'12, 14','2.4.6':'13–14','2.4.7':'15',
 '3.1.1':'1–2','3.1.2':'4','3.1.3':'3–6','3.1.4':'7–9','3.1.5':'12','3.1.6':'7','3.1.7':'8','3.1.8':'10','3.1.9':'6','3.1.10':'11, 13–15',
 '3.2.1':'1–2','3.2.2':'3–4','3.2.3':'5','3.2.4':'4–6','3.2.5':'6','3.2.6':'7','3.2.7':'8','3.2.8':'9','3.2.9':'10–11','3.2.10':'12','3.2.11':'13–15',
 '3.3.1':'1–3, 11','3.3.2':'3–4','3.3.3':'5','3.3.4':'7','3.3.5':'8','3.3.6':'9–11','3.3.7':'4, 6, 8','3.3.8':'2, 12','3.3.9':'13','3.3.10':'14–15'}


def paragraph(text='',style='Normal',bold=False):
    rpr='<w:rPr><w:b/></w:rPr>' if bold else ''
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def table(headers,rows):
    width=9300//len(headers)
    out=['<w:tbl><w:tblPr><w:tblW w:w="9300" w:type="dxa"/><w:tblBorders>']
    for name in ('top','left','bottom','right','insideH','insideV'):out.append(f'<w:{name} w:val="single" w:sz="4" w:color="B5C6D5"/>')
    out+=['</w:tblBorders></w:tblPr><w:tblGrid>',*(f'<w:gridCol w:w="{width}"/>' for _ in headers),'</w:tblGrid>']
    for rownum,row in enumerate([headers,*rows]):
        out.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if rownum==0 else '')+'</w:trPr>')
        for item in row:
            props=f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'+('<w:shd w:fill="E8F0F7"/>' if rownum==0 else '')+'</w:tcPr>'
            out.append('<w:tc>'+props+paragraph(str(item),'TableText',rownum==0)+'</w:tc>')
        out.append('</w:tr>')
    return ''.join(out)+'</w:tbl>'


class Book:
    def __init__(self):
        self.xml=[];self.files=[];self.marker=0;self.prose_chars=0;self.code_lines=0
    def p(self,text='',style='Normal'):
        self.xml.append(paragraph(text,style))
        if style not in ('Code','Command'):self.prose_chars+=len(text)
    def code(self,text,style='Code'):
        for line in text.split('\n'):self.p(line,style)
    def text(self,text):
        for line in text.split('\n'):
            if line.startswith('КОМАНДА: '):self.p(line[9:],'Command')
            else:self.p(line)
    def file(self,path,text,explanation='',split=True,roles=None):
        self.marker+=1;ident=str(self.marker);name=f'source_{self.marker:03}'
        digest=hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.p('Полный файл. Создайте его, затем добавляйте секции в указанном порядке.','Heading3')
        self.xml.append(f'<w:p><w:pPr><w:pStyle w:val="FilePath"/></w:pPr><w:bookmarkStart w:id="{ident}" w:name="{name}"/><w:r><w:t>{escape(path)}</w:t></w:r></w:p>')
        self.p('SHA-256: '+digest,'Small')
        if explanation:self.p(explanation)
        lines=text.split('\n');cuts={0:'Начало файла: описание, импорты и определения'}
        if split and path.endswith('.py'):
            t=ast.parse(text)
            for node in t.body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    start=min([node.lineno,*[d.lineno for d in node.decorator_list]])-1
                    cuts[start]=node.name
                    if isinstance(node,ast.ClassDef):
                        for member in node.body:
                            if isinstance(member,(ast.FunctionDef,ast.AsyncFunctionDef)):
                                begin=min([member.lineno,*[d.lineno for d in member.decorator_list]])-1
                                cuts[begin]=node.name+'.'+member.name
        positions=sorted(cuts)
        for i,start in enumerate(positions):
            end=positions[i+1] if i+1<len(positions) else len(lines)
            self.p(f'Секция {i+1}. {cuts[start]}','ListingHeading')
            annotation=(roles or {}).get(cuts[start].split('.')[-1])
            if annotation:self.p(annotation)
            for line in lines[start:end]:self.p(line,'Code');self.code_lines+=1
        self.xml.append(f'<w:p><w:pPr><w:pStyle w:val="Small"/></w:pPr><w:bookmarkEnd w:id="{ident}"/><w:r><w:t>Конец полного файла.</w:t></w:r></w:p>')
        self.files.append(dict(path=path,sha256=digest,lines=len(lines),bookmark=name))
    def save(self,path,title):
        styles=XML+f'''<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="24"/><w:lang w:val="ru-RU"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="140" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'''
        specs=[('Normal',24,None,None),('Title',44,'16324F',True),('Subtitle',26,'526578',None),('Heading1',34,'16324F',True),('Heading2',29,'176B9B',True),('Heading3',25,'176B9B',True),('ListingHeading',22,'526578',True),('FilePath',22,'16324F',True),('Small',19,'526578',None),('TableText',20,None,None),('Code',20,None,None),('Command',19,'16324F',None)]
        for name,size,color,bold in specs:
            iscode=name in ('Code','Command')
            pp=('<w:keepNext/>' if name in ('Title','Subtitle','Heading1','Heading2','Heading3','FilePath','ListingHeading') else '')
            if name=='Heading1':pp+='<w:pageBreakBefore/>'
            if name.startswith('Heading'):pp+=f'<w:outlineLvl w:val="{int(name[-1])-1}"/>'
            pp+=('<w:spacing w:after="0" w:line="220" w:lineRule="auto"/><w:shd w:fill="F4F6F8"/>' if iscode else '<w:spacing w:before="100" w:after="120"/>')
            if iscode:pp+='<w:keepLines/>'
            if name=='Normal':pp+='<w:jc w:val="both"/>'
            font='Consolas' if iscode else 'Calibri'
            rp=f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}"/><w:sz w:val="{size}"/>'
            if color:rp+=f'<w:color w:val="{color}"/>'
            if bold:rp+='<w:b/>'
            styles+=f'<w:style w:type="paragraph" w:styleId="{name}"'+(' w:default="1"' if name=='Normal' else '')+f'><w:name w:val="{name}"/><w:pPr>{pp}</w:pPr><w:rPr>{rp}</w:rPr></w:style>'
        styles+='</w:styles>'
        body=''.join(self.xml)+f'<w:sectPr><w:footerReference w:type="default" r:id="footer"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="992" w:bottom="1134" w:left="1417"/></w:sectPr>'
        document=XML+f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>'
        relationships=lambda data: XML+'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{ident}" Type="{R}/{typ}" Target="{target}"/>' for ident,typ,target in data)+'</Relationships>'
        footer=XML+f'<w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t>PredictML · </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'
        types=XML+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
        for name,typ in [('document','document.main'),('styles','styles'),('footer1','footer')]:types+=f'<Override PartName="/word/{name}.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.{typ}+xml"/>'
        types+='</Types>'
        files={'[Content_Types].xml':types,'_rels/.rels':relationships([('office','officeDocument','word/document.xml')]),'word/document.xml':document,'word/styles.xml':styles,'word/footer1.xml':footer,'word/_rels/document.xml.rels':relationships([('styles','styles','styles.xml'),('footer','footer','footer1.xml')])}
        path.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(path,'w',ZIP_DEFLATED) as z:
            for name,value in files.items():z.writestr(name,value.encode('utf-8'))
        return dict(title=title,path=str(path.relative_to(ROOT)),files=self.files,prose_chars=self.prose_chars,code_lines=self.code_lines)


def extract(book,output):
    output=Path(output)
    with ZipFile(book) as z:body=ET.fromstring(z.read('word/document.xml')).find(f'{{{W}}}body')
    active=None;results=[]
    for p in body.iter(f'{{{W}}}p'):
        start=p.find(f'{{{W}}}bookmarkStart')
        if start is not None and start.get(f'{{{W}}}name','').startswith('source_'):
            if active:raise ValueError('nested file listings')
            active=dict(id=start.get(f'{{{W}}}id'),lines=[])
        if not active:continue
        text=''.join(t.text or '' for t in p.iter(f'{{{W}}}t'))
        s=p.find(f'{{{W}}}pPr/{{{W}}}pStyle');style=s.get(f'{{{W}}}val') if s is not None else ''
        if style=='FilePath':active['path']=text
        if style=='Small' and text.startswith('SHA-256: '):active['hash']=text[9:]
        if style=='Code':active['lines'].append(text)
        end=p.find(f'{{{W}}}bookmarkEnd')
        if end is not None and end.get(f'{{{W}}}id')==active['id']:
            content='\n'.join(active['lines']).encode('utf-8')
            if hashlib.sha256(content).hexdigest()!=active['hash']:raise ValueError('listing checksum mismatch: '+active['path'])
            path=Path(active['path'])
            if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe listing path')
            target=output/path;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=content:raise ValueError('refusing overwrite: '+str(target))
            target.write_bytes(content);results.append(str(path));active=None
    if active:raise ValueError('unclosed listing')
    if not results:raise ValueError('no full source files in book')
    return results


def fmt(value):
    try:return f'{float(value):.6g}'
    except (ValueError,TypeError):return str(value)


def build_alignment(alignment):
    theory_path=COURSE/'source/theory-alignment.json'
    theory=json.loads(theory_path.read_text()) if theory_path.exists() else None
    base=[r for r in alignment if r['module']!=1]
    mapping=Book()
    mapping.p('Пособия, презентации и контроль','Title')
    mapping.p('Карта всех трёх модулей. Пособия содержат основной текст; в модуле 2 находятся полные листинги. Презентации сопровождают конкретные разделы. Нумерация разделов исходного пособия 1 отличается от нумерации презентаций: ориентируйтесь на заголовки, а не только на одинаковые цифры.')
    if theory:
        mapping.p('Модуль 1. Теория: вывод сопоставления','Heading1')
        mapping.p(theory['conclusion'])
        mapping.p('Тема 1 книги → презентации 1.1 и 1.2; темы 2 и 3 → презентация 1.3. Сквозные примеры, словарь и повторение используются при чтении. Это содержательное соответствие с разным делением, не буквальная копия оглавления.')
        for deck in ('1.1','1.2','1.3'):
            mapping.p('Презентация '+deck+' · 15 слайдов','Heading2')
            mapping.p('Контроль: тест '+deck+', 15 вопросов. Подробные комментарии и обратная карта — в отдельном отчёте теории.')
            rows=[r for r in theory['slides'] if r['presentation']==deck]
            mapping.xml.append(table(['Слайд','Тема слайда','Разделы пособия','Оценка'],[[r['slide'],r['title'],'; '.join(x['title'] for x in r['book_sections']),r['status']] for r in rows]))
        mapping.p('Точки согласования теории','Heading2')
        for item in theory['issues']:
            mapping.p(item['where'],'Heading3');mapping.p(item['finding']);mapping.p('Рекомендация: '+item['recommendation'])
        report=Book();report.p('Модуль 1. Пособие и презентации','Title');report.p(theory['conclusion']);report.p(theory['method'],'Small')
        report.p('Как читать карту','Heading1')
        report.p('Номера разделов берутся из исходного пособия. Например, раздел книги 1.3 о регрессии и презентация 1.3 о MLP — разные единицы. Номера абзацев служат устойчивыми указателями внутри проверенного DOCX, а не номерами страниц. Оригиналы не редактировались. Разная глубина сама по себе не является ошибкой: слайды дополняют чтение, а не заменяют весь текст.')
        report.xml.append(table(['Файл','SHA-256'],[[v['name'],v['sha256']] for v in theory['sources']]))
        for deck in ('1.1','1.2','1.3'):
            report.p('Презентация '+deck+': послайдовая карта','Heading1')
            for row in [r for r in theory['slides'] if r['presentation']==deck]:
                report.p(str(row['slide'])+'. '+row['title'],'Heading2')
                report.p('Пособие: '+'; '.join(v['title']+' (абзац '+str(v['paragraph'])+')' for v in row['book_sections']))
                report.p('Оценка: '+row['status']+'. '+row['comment'])
        report.p('Обратная карта: разделы пособия → слайды','Heading1')
        report.p('Пустая прямая ссылка означает отсутствие отдельного закреплённого слайда, а не обязательное отсутствие всех связанных понятий. Структурные заголовки, словарь и итоговый контроль естественно могут оставаться в книге.')
        report.xml.append(table(['Раздел книги','Слайды: презентация/номер','Комментарий'],[[r['title'],', '.join(r['slides']) or 'Отдельной ссылки нет',r['status']] for r in theory['book_sections']]))
        report.p('Различия и предложения','Heading1')
        for item in theory['issues']:
            report.p(item['where'],'Heading2');report.p('Тип: '+item['priority']+'. '+item['finding']);report.p('Рекомендация: '+item['recommendation'])
        report.p('Общий итог','Heading1');report.p('Основной теоретический материал можно использовать совместно с тремя презентациями, если задан маршрут чтения. До единой финальной редакции курса следует согласовать пример цели спроса и распределение реализации между модулями. Редакторские изменения авторских документов этим отчётом не выполнены. При требовании подробного объяснения всей темы обучения на экране следует отдельно рассмотреть увеличение числа презентаций, сохраняя полноту пособия.')
        report.save(COURSE/'manuals/Сопоставление_модуля_1_теория.docx','Сопоставление теории')
    practice={
        '2.1':'Чтение таблицы, явная схема, fit/transform и проверки входа.',
        '2.2':'Три алгоритма на одних строках, лучший checkpoint и отсутствие утечки.',
        '2.3':'Спрос, закупка, цена, сохранение и повторная загрузка модели.',
        '2.4':'Установка, пять учебных проверок, полный CLI-цикл и запуск интерфейса.',
        '3.1':'Паспорт данных, фиксированный план и обучение кандидатов.',
        '3.2':'Новые строки, выбранная версия, трёхмесячный спрос и закупка.',
        '3.3':'Общий контроль, серия seed, разбор ошибок и итоговый отчёт.'}
    for deck in practice:
        mapping.p('Презентация '+deck+' · 15 слайдов','Heading1');mapping.p('Практика: '+practice[deck]+' Контроль: тест '+deck+', 15 вопросов.')
        rows=[r for r in base if r['presentation']==deck]
        mapping.xml.append(table(['Подраздел пособия','Тема','Слайды'],[[r['section'],r['title'],r['slides']] for r in rows]))
    mapping.p('Итоговый контроль курса','Heading1');mapping.p('Отдельный итоговый тест содержит 30 вопросов: по 10 на теорию, разработку и обучение/эксплуатацию. В каждом вопросе три варианта и один правильный ответ. Порог зачёта и время выполнения определяются отдельно; ключи хранятся в преподавательском файле.')
    mapping.save(COURSE/'manuals/Сопоставление_пособий_и_презентаций.docx','Карта соответствия всех модулей')
    combined=(theory['slides'] if theory else [])+base
    (COURSE/'source/manual-slide-map.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--alignment-only',action='store_true');parser.add_argument('--extract');parser.add_argument('--output');args=parser.parse_args()
    if args.extract:
        if not args.output:parser.error('--output required')
        print(json.dumps(extract(args.extract,args.output),ensure_ascii=False,indent=2));return
    if args.alignment_only:
        build_alignment(json.loads((COURSE/'source/manual-slide-map.json').read_text()))
        print('Rebuilt theory report and all-module alignment; full manuals unchanged.')
        return
    content=json.loads((COURSE/'source/manuals-content.json').read_text())
    notes=json.loads((COURSE/'source/manual-code-notes.json').read_text())
    reference=json.loads((COURSE/'source/experiment-reference.json').read_text())
    for path,digest in reference['runtime_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('repeat reference experiment for changed '+path)
    manifest=[];alignment=[]
    for spec in content:
        b=Book();module=spec['module'];b.p(f'Модуль {module}','Subtitle');b.p(spec['title'],'Title');b.p('Полное учебное пособие. Код, последовательность действий, практикум и контроль результата.','Subtitle');b.p('Редакция 1.0 · 23 ноября 2026 года','Small')
        for text in spec['intro']:b.p(text)
        b.p('Содержание и связь с презентациями','Heading1')
        b.xml.append(table(['Раздел','Презентация','Практический результат'],[[s['id']+' '+s['title'],s['slides'],'Полные листинги и проверка' if module==2 else 'Воспроизводимый опыт и отчёт'] for s in spec['sections']]))
        if module==3:
            b.p('Среда справочного эксперимента','Heading2');b.xml.append(table(['Компонент','Версия'],[['Python',reference['environment']['python']],*reference['environment']['packages'].items(),['PyTorch threads',reference['torch_threads']]]))
        for s in spec['sections']:
            b.p(s['id']+'. '+s['title'],'Heading1');b.p('Сопровождение: презентация '+s['slides']+'. Тест: '+s['id']+'.','Small')
            inserted=False;in_code=False
            def insert_listings():
                for short in ORDER[s['id']]:
                    path='predictml/'+short;b.file(path,(ROOT/path).read_text(),notes['files'][path],roles=notes['functions'])
            for text in s['paragraphs']:
                heading=re.match(r'^(\d\.\d\.\d+)\. ',text)
                if module==2 and heading and heading[1] in ('2.1.7','2.2.6','2.3.7','2.4.5') and not inserted:
                    b.p('Последовательная сборка файлов раздела','Heading2');insert_listings();inserted=True
                if module==3 and heading:
                    key=heading[1]
                    if key=='3.3.3':
                        b.xml.append(table(['Алгоритм','Часть','MAE','RMSE','wMAPE, %','R²'],[[r['algorithm'],r['partition'],fmt(r['mae']),fmt(r['rmse']),fmt(r['wmape']),fmt(r['r2'])] for r in reference['internal']]))
                    if key=='3.3.5':
                        b.xml.append(table(['Seed весов','Seed split','Val MAE','Test MAE','Best epoch','Epochs ran'],[[r['initialization_seed'],r['split_seed'],fmt(r['validation_mae']),fmt(r['test_mae']),r['best_epoch'],r['epochs_ran']] for r in reference['seeds']]))
                    if key=='3.3.8':
                        b.xml.append(table(['Вариант','Control MAE','Control RMSE','Control R²'],[[r['variant'],fmt(r['mae']),fmt(r['rmse']),fmt(r['r2'])] for r in reference['control']]))
                    if key=='3.3.9':
                        b.xml.append(table(['Задача','Алгоритм','Часть','MAE','RMSE'],[[r['task'],r['algorithm'],r['partition'],fmt(r['mae']),fmt(r['rmse'])] for r in reference['scenarios']]))
                if heading:
                    b.p(text,'Heading2');code=heading[1];slides=SLIDE_MAP[code];deck='.'.join(code.split('.')[:2]);b.p(f'Презентация {deck}, слайды {slides}.','Small');alignment.append(dict(module=module,section=code,title=text,presentation=deck,slides=slides,test=deck));continue
                for line in text.split('\n'):
                    if line=='КОД:':in_code=True;continue
                    if line=='КОНЕЦ КОДА':in_code=False;continue
                    if in_code:b.p(line,'Code')
                    elif line.startswith('КОМАНДА: '):b.p(line[9:],'Command')
                    else:b.p(line)
            if module==2 and not inserted:insert_listings()
        if module==2:
            b.p('Приложение А. Окружение и упаковка','Heading1')
            b.p('Создайте эти файлы в корне учебного проекта до установки. Учебный README и MANIFEST не копируют историю авторского репозитория; исполняемый код и контракт пакета совпадают. Все runtime-файлы ниже восстанавливаются без внешнего репозитория.')
            for filename in ['requirements.txt','requirements-dev.txt','requirements-browser.txt','pyproject.toml']:
                b.file(filename,(ROOT/filename).read_text(),split=False)
            b.file('README.md','# PredictML — учебный проект\n\nПолный код восстановлен по модулю 2.\n\nЗапуск: python -m predictml ui\nПримеры: python -m predictml demo --output examples\nПроверка: python tools/check_program.py\nПолный цикл: python tools/full_cycle.py --output results/check\n',split=False)
            b.file('MANIFEST.in','include README.md requirements.txt requirements-dev.txt requirements-browser.txt pyproject.toml\nrecursive-include tools *.py\n',split=False)
            b.file('.gitignore','.venv/\n__pycache__/\n*.pyc\n*.egg-info/\nbuild/\ndist/\nresults/\nexamples/\n',split=False)
            b.p('Приложение Б. Полные исходные данные','Heading1');b.p('Каждый следующий листинг является содержимым указанного UTF-8-файла целиком. Это фиксированные синтетические наблюдения. Housing.xlsx создаётся инструментом следующего приложения; отдельный бинарный файл из авторской папки не требуется.')
            for path in sorted((ROOT/'predictml/data/examples').rglob('*')):
                if path.is_file() and path.suffix!='.xlsx':b.file(str(path.relative_to(ROOT)),path.read_text(),split=False)
            b.p('Приложение В. Инструменты проверки','Heading1')
            for path,origin in [('tools/create_workbook.py','docs/course/practice/create_workbook.py'),('tools/check_program.py','docs/course/practice/check_program.py'),('tools/full_cycle.py','scripts/acceptance.py')]:b.file(path,(ROOT/origin).read_text(),split=True)
        else:
            b.p('Приложение А. Полный скрипт эксперимента','Heading1');b.p('Создайте tools/run_experiment.py. Он использует программу модуля 2, заранее фиксирует условия и сохраняет все результаты. Не изменяйте бюджет по контрольной MAE в ходе повторения справочного опыта.')
            b.file('tools/run_experiment.py',(COURSE/'practice/run_experiment.py').read_text(),split=True)
            b.p('Приложение Б. Форма отчёта','Heading1')
            for title,description in [('Паспорт','Источник, дата/версия, строки, столбцы, единицы, период и ограничения.'),('План','Проверяемый вопрос, split, метрика рекомендации, отдельный контроль, бюджет и список seed.'),('Исполнение','Окружение, команды, ID моделей, путь к входам и результатам.'),('Результаты','Validation и test отдельно, общий контроль версий, разброс seed, by_lead спроса.'),('Ошибки','Две характерные строки, знак и величина ошибки, происхождение пропуска или смещения.'),('Решение','Выбор по принятому критерию либо отказ от обновления с численным обоснованием.'),('Границы','Синтетика, одно временное окно, экстраполяция, формульная цель и следующий новый опыт.')]:b.p(title,'Heading2');b.p(description)
            b.p('Не заполняйте форму универсальными утверждениями о качестве ML. Каждое решение должно ссылаться на сопоставимые наблюдения и известную роль набора. Скриншот графика допускается как иллюстрация, но не заменяет данные и условия опыта.')
        filename='Модуль_2_Разработка_программы.docx' if module==2 else 'Модуль_3_Обучение_и_эксплуатация.docx'
        manifest.append(b.save(COURSE/'manuals'/filename,spec['title']))
    build_alignment(alignment)
    (COURSE/'source/manuals-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in m.items() if k!='files'}|{'complete_listings':len(m['files'])} for m in manifest],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
