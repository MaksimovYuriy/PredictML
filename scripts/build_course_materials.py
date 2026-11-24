"""Build course PPTX and DOCX from reviewed JSON; Pillow is used for text measurement.

Usage: python scripts/build_course_materials.py
The author's original theory presentations are not modified.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import json
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT/'docs/course'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
NAVY, BLUE, GREY, PALE = '16324F', '176B9B', '526578', 'EEF5FA'


def relationships(items):
    return XML+f'<Relationships xmlns="{REL}">'+''.join(
        f'<Relationship Id="{ident}" Type="{R}/{kind}" Target="{target}"/>'
        for ident, kind, target in items)+'</Relationships>'


def package(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value.encode('utf-8') if isinstance(value, str) else value)


def group():
    return '<p:nvGrpSpPr><p:cNvPr id="1" name="Group"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'


def box(ident, x, y, w, h, paragraphs, *, size=20, color=NAVY, fill=None, bold=False):
    transform=f'<a:xfrm><a:off x="{int(x*914400)}" y="{int(y*914400)}"/><a:ext cx="{int(w*914400)}" cy="{int(h*914400)}"/></a:xfrm>'
    background=f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    shape=f'<p:sp><p:nvSpPr><p:cNvPr id="{ident}" name="Text {ident}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>{transform}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{background}<a:ln><a:noFill/></a:ln></p:spPr>'
    text=''
    for value in paragraphs:
        pieces=value.split('\n')
        run=f'<a:rPr lang="ru-RU" sz="{size*100}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Calibri"/><a:cs typeface="Calibri"/></a:rPr>'
        runs='<a:br/>'.join(f'<a:r>{run}<a:t xml:space="preserve">{escape(part)}</a:t></a:r>' for part in pieces)
        text+=f'<a:p><a:pPr><a:lnSpc><a:spcPct val="110000"/></a:lnSpc><a:spcAft><a:spcPts val="1000"/></a:spcAft></a:pPr>{runs}<a:endParaRPr lang="ru-RU" sz="{size*100}"/></a:p>'
    return shape+f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>{text}</p:txBody></p:sp>'


def wrapped(value, width, size=20, bold=False):
    # Measure conservatively in DejaVu; Office Calibri is usually narrower.
    names = (['DejaVuSans-Bold.ttf', 'C:/Windows/Fonts/calibrib.ttf'] if bold
             else ['DejaVuSans.ttf', 'C:/Windows/Fonts/calibri.ttf'])
    font = None
    for name in names:
        try:
            font = ImageFont.truetype(name, round(size*96/72))
            break
        except OSError:
            continue
    def measure(text):
        return font.getlength(text) if font else len(text)*size*96/72*.8
    limit = width*96-8
    result = []
    for original in value.split('\n'):
        indent = original[:len(original)-len(original.lstrip())]
        line = ''
        for word in original.split():
            if measure(indent+(line+' '+word).strip()) > limit and line:
                result.append(indent+line)
                line = ''
            while measure(word) > limit:
                cut = 1
                while cut < len(word) and measure(word[:cut+1]) <= limit:
                    cut += 1
                result.append(word[:cut])
                word = word[cut:]
            line = (line+' '+word).strip()
        result.append(indent+line)
    return '\n'.join(result)


def slide_shapes(deck, slide, number):
    shapes=[]
    shapes.append(box(2, .55, .26, 12.2, .35, [f'МОДУЛЬ {deck["id"][0]}  /  ПРЕЗЕНТАЦИЯ {deck["id"]}'], size=12, color=BLUE, bold=True))
    shapes.append(box(3, .55, .88, 12.15, 1.1, [wrapped(slide['title'], 12.15, 29, True)], size=29, bold=True))
    points=[wrapped('• '+point, 6.0, 20) for point in slide['points']]
    shapes.append(box(4, .63, 2.25, 6.0, 4.38, points, size=20))
    shapes.append(box(5, 7.05, 2.22, 5.7, 4.15, [], fill=PALE))
    shapes.append(box(6, 7.32, 2.5, 5.08, .6, [slide['example_label'].upper()], size=13, color=BLUE, bold=True))
    example=wrapped(slide['example'], 5.08, 19)
    shapes.append(box(7, 7.32, 3.28, 5.08, 2.75, [example], size=19))
    shapes.append(box(8, .6, 7.02, 11.2, .25, [deck['title']], size=10, color=GREY))
    shapes.append(box(9, 11.9, 7.0, .8, .3, [f'{number:02d} / 15'], size=11, color=BLUE))
    return ''.join(shapes)


def make_pptx(deck, path):
    files={}
    files['_rels/.rels']=relationships([('rId1','officeDocument','ppt/presentation.xml')])
    ids=''.join(f'<p:sldId id="{255+i}" r:id="slide{i}"/>' for i in range(1,16))
    files['ppt/presentation.xml']=XML+f'<p:presentation xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="master"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="notesMaster"/></p:notesMasterIdLst><p:sldIdLst>{ids}</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    files['ppt/_rels/presentation.xml.rels']=relationships([('master','slideMaster','slideMasters/slideMaster1.xml'),('notesMaster','notesMaster','notesMasters/notesMaster1.xml')]+[(f'slide{i}','slide',f'slides/slide{i}.xml') for i in range(1,16)])
    color_map='<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
    files['ppt/slideMasters/slideMaster1.xml']=XML+f'<p:sldMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group()}</p:spTree></p:cSld>{color_map}<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="layout"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    files['ppt/slideMasters/_rels/slideMaster1.xml.rels']=relationships([('layout','slideLayout','../slideLayouts/slideLayout1.xml'),('theme','theme','../theme/theme1.xml')])
    files['ppt/slideLayouts/slideLayout1.xml']=XML+f'<p:sldLayout xmlns:p="{P}" xmlns:a="{A}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    files['ppt/slideLayouts/_rels/slideLayout1.xml.rels']=relationships([('master','slideMaster','../slideMasters/slideMaster1.xml')])
    colors=dict(dk1=NAVY,lt1='FFFFFF',dk2=GREY,lt2=PALE,accent1=BLUE,accent2='20A39E',accent3='E6A23C',accent4='8A69B0',accent5='78909C',accent6='C06E52',hlink='176B9B',folHlink='7E57A0')
    clr=''.join(f'<a:{name}><a:srgbClr val="{value}"/></a:{name}>' for name,value in colors.items())
    font='<a:fontScheme name="Course"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface="Calibri"/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface="Calibri"/></a:minorFont></a:fontScheme>'
    fill='<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    line='<a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'
    fmt=f'<a:fmtScheme name="Course"><a:fillStyleLst>{fill*3}</a:fillStyleLst><a:lnStyleLst>{line*3}</a:lnStyleLst><a:effectStyleLst>{"<a:effectStyle><a:effectLst/></a:effectStyle>"*3}</a:effectStyleLst><a:bgFillStyleLst>{fill*3}</a:bgFillStyleLst></a:fmtScheme>'
    files['ppt/theme/theme1.xml']=XML+f'<a:theme xmlns:a="{A}" name="PredictML"><a:themeElements><a:clrScheme name="Course">{clr}</a:clrScheme>{font}{fmt}</a:themeElements></a:theme>'
    files['ppt/notesMasters/notesMaster1.xml']=XML+f'<p:notesMaster xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{group()}</p:spTree></p:cSld>{color_map}<p:notesStyle/></p:notesMaster>'
    files['ppt/notesMasters/_rels/notesMaster1.xml.rels']=relationships([('theme','theme','../theme/theme1.xml')])
    for i,slide in enumerate(deck['slides'],1):
        files[f'ppt/slides/slide{i}.xml']=XML+f'<p:sld xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill></p:bgPr></p:bg><p:spTree>{group()}{slide_shapes(deck,slide,i)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
        files[f'ppt/slides/_rels/slide{i}.xml.rels']=relationships([('layout','slideLayout','../slideLayouts/slideLayout1.xml'),('notes','notesSlide',f'../notesSlides/notesSlide{i}.xml')])
        notes=box(2,.5,.5,6.3,8,[f'{deck["id"]}, слайд {i}. '+slide['notes']],size=12)
        # Mark the notes text as body content for the PowerPoint notes pane.
        notes=notes.replace('<p:nvPr/>','<p:nvPr><p:ph type="body" idx="1"/></p:nvPr>',1)
        files[f'ppt/notesSlides/notesSlide{i}.xml']=XML+f'<p:notes xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{group()}{notes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>'
        files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=relationships([('master','notesMaster','../notesMasters/notesMaster1.xml'),('slide','slide',f'../slides/slide{i}.xml')])
    types={'ppt/presentation.xml':'presentation.main','ppt/slideMasters/slideMaster1.xml':'slideMaster','ppt/slideLayouts/slideLayout1.xml':'slideLayout','ppt/notesMasters/notesMaster1.xml':'notesMaster'}
    for i in range(1,16):
        types[f'ppt/slides/slide{i}.xml']='slide';types[f'ppt/notesSlides/notesSlide{i}.xml']='notesSlide'
    overrides=''.join(f'<Override PartName="/{name}" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml"/>' for name,kind in types.items())
    files['[Content_Types].xml']=XML+f'<Types xmlns="{CT}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>{overrides}<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>'
    package(path,files)


W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def para(text, *, style=None, bold=False):
    properties=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
    runprops='<w:rPr><w:b/></w:rPr>' if bold else ''
    parts='<w:br/>'.join(f'<w:t xml:space="preserve">{escape(v)}</w:t>' for v in text.split('\n'))
    return f'<w:p>{properties}<w:r>{runprops}{parts}</w:r></w:p>'


def docx(path, paragraphs):
    styles=f'''<w:styles xmlns:w="{W}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="24"/><w:lang w:val="ru-RU"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="260" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:color w:val="{NAVY}"/><w:sz w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="140"/></w:pPr><w:rPr><w:b/><w:color w:val="{BLUE}"/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Question"><w:name w:val="Question"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="180" w:after="80"/></w:pPr><w:rPr><w:b/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Option"><w:name w:val="Option"/><w:basedOn w:val="Normal"/><w:pPr><w:keepLines/><w:ind w:left="280"/><w:spacing w:after="50"/></w:pPr></w:style><w:style w:type="paragraph" w:styleId="OptionKeep"><w:name w:val="OptionKeep"/><w:basedOn w:val="Option"/><w:pPr><w:keepNext/></w:pPr></w:style></w:styles>'''
    document=XML+f'<w:document xmlns:w="{W}"><w:body>'+''.join(paragraphs)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1417"/></w:sectPr></w:body></w:document>'
    files={'_rels/.rels':relationships([('document','officeDocument','word/document.xml')]),'word/document.xml':document,'word/styles.xml':XML+styles,'word/_rels/document.xml.rels':relationships([('styles','styles','styles.xml')]),'[Content_Types].xml':XML+f'<Types xmlns="{CT}"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'}
    package(path,files)


def make_tests(module, packs):
    key=[para(f'Модуль {module}. Ответы и пояснения',style='Title'),para('Преподавательская версия. В каждом вопросе ровно один правильный ответ. Каждое задание — 1 балл; максимум за отдельный тест — 15. Порог зачёта устанавливается отдельно.')]
    questions_md=[f'# Модуль {module}. Тестовые вопросы\n','В каждом вопросе один правильный ответ из трёх.\n']
    answers_md=[f'# Модуль {module}. Ответы и пояснения\n']
    for pack in packs:
        ident=pack['presentation'];title=pack['title']
        student=[para(f'Тест {ident}. {title}',style='Title'),para('ФИО: ____________________  Дата: __________'),para('15 вопросов среднего уровня. Выберите один правильный ответ в каждом вопросе. Максимум — 15 баллов. Ответы запишите буквами А, Б или В.')]
        key.append(para(f'{ident}. {title}',style='Heading1'));questions_md.append(f'\n## {ident}. {title}\n');answers_md.append(f'\n## {ident}. {title}\n')
        for i,q in enumerate(pack['questions'],1):
            student.append(para(f'{i}. {q["question"]}',style='Question'));questions_md.append(f'\n{i}. {q["question"]}\n')
            for j,option in enumerate(q['options']):
                student.append(para(f'{"АБВ"[j]}) {option}',style='OptionKeep' if j<2 else 'Option'));questions_md.append(f'   {"АБВ"[j]}) {option}\n')
            answer='АБВ'[q['correct_index']]
            key.append(para(f'{i}. {answer} — {q["options"][q["correct_index"]]}',bold=True));key.append(para(f'Пояснение: {q["explanation"]}'))
            key.append(para(f'Источник: презентация {ident}, слайд {q["slide"]}'+(' — '+q['source_title'] if 'source_title' in q else '')+'.'))
            answers_md.append(f'{i}. **{answer}** — {q["explanation"]} Источник: презентация {ident}, слайд {q["slide"]}.\n')
        docx(COURSE/f'tests/Тест_{ident.replace(".","_")}.docx',student)
    docx(COURSE/f'tests/Ключи_Модуль_{module}.docx',key)
    (COURSE/f'tests/module-{module}-questions.md').write_text('\n'.join(questions_md)+'\n')
    (COURSE/f'tests/module-{module}-answers.md').write_text('\n'.join(answers_md)+'\n')


def make_final_test():
    path=COURSE/'tests/final-course-bank.json'
    if not path.exists():
        return
    data=json.loads(path.read_text());rows=data['questions']
    assert len(rows)==30
    assert [sum(q['module']==m for q in rows) for m in (1,2,3)]==[10,10,10]
    assert [sum(q['correct_index']==i for q in rows) for i in range(3)]==[10,10,10]
    student=[para(data['title'],style='Title'),para('ФИО: ____________________  Дата: __________'),para(data['instructions']),para('Проверяются теория, разработка и обучение/эксплуатация. Каждый вопрос оценивается в 1 балл. Время и порог зачёта устанавливаются отдельно.')]
    teacher=[para('Ключи. '+data['title'],style='Title'),para('Только для преподавателя. 30 вопросов, по 10 на каждый модуль. Один правильный ответ; максимум 30 баллов. Средний уровень — авторская оценка, а не результат статистической апробации.')]
    teacher.append(para('Краткий ключ',style='Heading1'))
    for start in (0,10,20):teacher.append(para('   '.join(f'{i+1} — {"АБВ"[q["correct_index"]]}' for i,q in enumerate(rows[start:start+10],start))))
    question_text=['# '+data['title'],data['instructions']]
    answer_text=['# Итоговый тест. Ответы и пояснения','Преподавательская версия. Максимум 30 баллов.']
    for i,q in enumerate(rows,1):
        assert len(set(q['options']))==3
        if i in (1,11,21):
            name={1:'Теория',2:'Разработка',3:'Обучение и эксплуатация'}[q['module']]
            student.append(para(f'Модуль {q["module"]}. {name}',style='Heading1'))
            teacher.append(para(f'Модуль {q["module"]}. {name}',style='Heading1'))
            question_text.append(f'\n## Модуль {q["module"]}. {name}\n')
        student.append(para(f'{i}. {q["question"]}',style='Question'))
        question_text.append(f'\n{i}. {q["question"]}\n')
        for j,option in enumerate(q['options']):
            student.append(para(f'{"АБВ"[j]}) {option}',style='OptionKeep' if j<2 else 'Option'))
            question_text.append(f'   {"АБВ"[j]}) {option}\n')
        letter='АБВ'[q['correct_index']]
        teacher.append(para(f'{i}. {letter} — {q["options"][q["correct_index"]]}',bold=True))
        teacher.append(para('Пояснение: '+q['explanation']))
        sources='; '.join(f'Модуль {r["module"]}, пособие §{r["manual_section"]}; презентация {r["presentation"]}, слайды '+', '.join(map(str,r['slides'])) for r in q['sources'])
        teacher.append(para('Источник: '+sources+'.'))
        answer_text.append(f'{i}. **{letter}** — {q["explanation"]} Источник: {sources}.\n')
    student.append(para('Лист ответов',style='Heading1'))
    for start in (1,11,21):student.append(para('   '.join(f'{i}: ____' for i in range(start,start+10))))
    docx(COURSE/'tests/Итоговый_тест_30_вопросов.docx',student)
    docx(COURSE/'tests/Ключи_Итоговый_тест.docx',teacher)
    (COURSE/'tests/final-course-questions.md').write_text('\n'.join(question_text)+'\n')
    (COURSE/'tests/final-course-answers.md').write_text('\n'.join(answer_text)+'\n')


def main():
    decks=json.loads((COURSE/'source/presentations.json').read_text())
    for deck in decks:
        assert len(deck['slides'])==15
        filename=f'Презентация_{deck["id"].replace(".","_")}.pptx'
        make_pptx(deck,COURSE/'presentations'/filename)
    for module in (1,2,3):
        packs=json.loads((COURSE/f'tests/module-{module}-bank.json').read_text())
        for pack in packs:
            assert len(pack['questions'])==15
            assert [sum(q['correct_index']==i for q in pack['questions']) for i in range(3)]==[5,5,5]
        make_tests(module,packs)
    make_final_test()
    print('Built 7 PPTX, 10 thematic tests, 3 module keys and the 30-question final test with keys.')


if __name__=='__main__':
    main()
