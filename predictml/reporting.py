"""Portable comparison reports: metrics CSV, prediction CSV and HTML scatter."""
import json
from pathlib import Path
import pandas as pd


def export_comparison(directory, models, rows, split, context=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = [dict(model=name, partition=part, fit_seconds=details.get('fit_seconds', details.get('training', {}).get('fit_seconds')), **details[part])
               for name, details in models.items() for part in ('validation', 'test')]
    pd.DataFrame(metrics).to_csv(directory/'comparison.csv', index=False)
    pd.DataFrame(rows).to_csv(directory/'comparison_predictions.csv', index=False)
    payload = json.dumps(dict(models=models, rows=rows, split=split, context=context or {}), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>PredictML: сравнение</title>
<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:16px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}svg{width:100%;max-width:650px;background:#f8fafc}select{font:inherit}</style>
<h1>Сравнение моделей</h1><p>Выбор модели — по validation. Test используется только для итоговой оценки.</p>
<div id="metrics"></div><p>Модель: <select id="model"></select></p><p>Test: по горизонтали — факт, по вертикали — прогноз. Диагональ означает точное совпадение.</p>
<svg id="plot" viewBox="0 0 640 480" role="img" aria-label="Факты и прогнозы"></svg><details><summary>Границы и состав выборок</summary><pre id="split"></pre></details>
<details><summary>Параметры запуска и единицы</summary><pre id="context"></pre></details>
<script>const data=PAYLOAD;const select=document.getElementById('model');
for(const name of Object.keys(data.models)){const o=document.createElement('option');o.textContent=name;select.append(o)}
const table=document.createElement('table');const header=table.insertRow();for(const t of ['Модель','Выборка','MAE','MSE','RMSE','wMAPE %','R²']){header.appendChild(document.createElement('th')).textContent=t}
for(const [name,parts] of Object.entries(data.models))for(const part of ['validation','test']){const r=table.insertRow();for(const v of [name,part,...['mae','mse','rmse','wmape','r2'].map(k=>parts[part][k])])r.insertCell().textContent=v===null?'не определено':typeof v==='number'?v.toPrecision(5):v}
document.getElementById('metrics').append(table);document.getElementById('split').textContent=JSON.stringify(data.split,null,2);document.getElementById('context').textContent=JSON.stringify(data.context,null,2);
function draw(){const svg=document.getElementById('plot');svg.replaceChildren();const rows=data.rows.filter(r=>r.model===select.value);const vals=rows.flatMap(r=>[r.actual,r.prediction]);const lo=Math.min(0,...vals),hi=Math.max(...vals),span=hi-lo||1;const x=v=>50+(v-lo)/span*540,y=v=>430-(v-lo)/span*380;const ns='http://www.w3.org/2000/svg';function el(tag,attrs){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);svg.append(e);return e}el('line',{x1:50,y1:430,x2:590,y2:50,stroke:'#999'});for(const r of rows){const e=el('circle',{cx:x(r.actual),cy:y(r.prediction),r:4,fill:'#2563eb',opacity:.7});const t=document.createElementNS(ns,'title');t.textContent=JSON.stringify(r);e.append(t)}for(const [value,xx,yy]of [[lo,50,455],[hi,570,455]]){el('text',{x:xx,y:yy}).textContent=value.toPrecision(4)}}select.onchange=draw;draw();</script></html>'''
    (directory/'comparison.html').write_text(html.replace('PAYLOAD', payload), encoding='utf-8')
