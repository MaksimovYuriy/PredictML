"""Run: python -m streamlit run predictml/ui/app.py"""
from pathlib import Path
import json
import os
import traceback
import streamlit as st
import pandas as pd
from predictml.config import TrainingConfig
from predictml.demo import examples_dir
from predictml.storage import Store
from predictml.ui.helpers import save_upload, save_group, layers, version_label


def guarded(action):
    try:
        with st.spinner('Выполняется…',show_time=True):
            return action()
    except Exception as exc:
        st.error(f'Не удалось выполнить операцию: {exc}')
        with st.expander('Подробности ошибки'):
            st.code(traceback.format_exc())
        return None


def table_input(key, workspace, *, demo=False):
    use_demo = st.checkbox('Синтетический пример', value=demo, key=key+'_demo')
    upload = st.file_uploader('Таблица CSV/XLSX', type=['csv','xlsx'], key=key+'_upload')
    if upload is not None:
        path = save_upload(upload, workspace)
    elif use_demo:
        path = examples_dir()/'tabular/housing.csv'
    else:
        return None, None, None
    from predictml.tabular.io import sheets, read_table
    sheet = st.selectbox('Лист', sheets(path), key=key+'_sheet') if path.suffix.lower()=='.xlsx' else None
    frame = guarded(lambda: read_table(path,sheet))
    if frame is not None:
        st.caption(f'{len(frame)} строк · {len(frame.columns)} столбцов')
        st.dataframe(frame.head(30), hide_index=True)
        with st.expander('Пропуски'):
            st.dataframe(frame.isna().sum().rename('Пропуски'))
    return (path, sheet, frame) if frame is not None else (None, sheet, None)


def training_options(key, fixed_layers=None):
    with st.expander('Настройки обучения', expanded=False):
        epochs = st.number_input('Максимум эпох',min_value=1,max_value=100000,value=100,key=key+'_epochs')
        if fixed_layers is None:
            hidden = st.text_input('Скрытые слои (через запятую)',value='32,16',key=key+'_layers')
        else:
            hidden = ','.join(str(v) for v in fixed_layers)
            st.caption('Сохранённые скрытые слои: '+hidden)
        rate = st.number_input('Learning rate',min_value=.000001,max_value=1.,value=.001,format='%.6f',key=key+'_lr')
        batch = st.number_input('Batch size',min_value=1,value=32,key=key+'_batch')
        seed = st.number_input('Seed',min_value=0,max_value=2**32-1,value=42,key=key+'_seed')
        stop = st.checkbox('Ранняя остановка',value=True,key=key+'_stop')
        patience = st.number_input('Проверок без улучшения',min_value=1,value=20,key=key+'_patience')
        delta = st.number_input('Минимальное улучшение',min_value=0.,value=0.,key=key+'_delta')
    return dict(epochs=int(epochs),learning_rate=float(rate),batch_size=int(batch),seed=int(seed),
                validation_interval=1,patience=int(patience) if stop else None,min_delta=float(delta)), hidden


def artifacts(directory):
    directory=Path(directory)
    if not directory.exists():
        st.warning('Каталог результатов не найден')
        return
    st.caption(f'Результаты: {directory}')
    csv=directory/'comparison_predictions.csv'
    if csv.exists():
        frame=pd.read_csv(csv)
        if 'model' in frame:
            chosen=st.selectbox('Модель на графике',frame.model.unique(),key=str(directory)+'_chart')
            shown=frame[frame.model==chosen]
        else:
            shown=frame
        if {'actual','prediction'} <= set(shown):
            st.scatter_chart(shown,x='actual',y='prediction',x_label='Факт',y_label='Прогноз')
        st.dataframe(shown,hide_index=True)
    elif (directory/'predictions.csv').exists():
        frame=pd.read_csv(directory/'predictions.csv')
        st.dataframe(frame,hide_index=True)
        if 'prediction' in frame:
            st.line_chart(frame['prediction'])
    for name in ('demand.csv','materials.csv','material_months.csv','prices.csv'):
        if (directory/name).exists():
            with st.expander(name,expanded=True):
                st.dataframe(pd.read_csv(directory/name),hide_index=True)
    metrics=directory/'comparison.csv'
    if metrics.exists():
        st.subheader('Метрики')
        st.caption('MAE/RMSE — исходные единицы, MSE — их квадрат, wMAPE — %. Пустые wMAPE/R² означают неопределённую метрику.')
        st.dataframe(pd.read_csv(metrics),hide_index=True)
    files=[p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.csv','.xlsx','.json','.html')]
    if files:
        st.subheader('Скачать результаты')
        for path in files:
            st.download_button(path.name,path.read_bytes(),file_name=path.name,key=str(path)+'_download')


def data_directory(key, workspace):
    mode=st.radio('Источник данных',['Загрузить файлы','Локальный каталог'],key=key+'_mode',horizontal=True)
    if mode=='Локальный каталог':
        value=st.text_input('Каталог подготовленных данных',key=key+'_directory')
        return Path(value).expanduser() if value else None
    uploads=st.file_uploader('CSV/JSON с исходными именами файлов',type=['csv','json'],accept_multiple_files=True,key=key+'_files')
    if not uploads:
        return None
    # Stable selection across reruns; group is saved only when content changes.
    import hashlib
    signature=tuple((f.name,hashlib.sha256(f.getvalue()).hexdigest()) for f in uploads)
    if st.session_state.get(key+'_signature')!=signature:
        st.session_state[key+'_directory_saved']=str(save_group(uploads,workspace))
        st.session_state[key+'_signature']=signature
    return Path(st.session_state[key+'_directory_saved'])


def train_page(store, workspace):
    st.header('Обучение по своей таблице')
    path,sheet,frame=table_input('train',workspace,demo=True)
    if frame is None:
        return
    signature=(str(path),sheet)
    st.caption('Цель должна быть числовой. Строки без цели исключаются; пропуски признаков обрабатываются по train.')
    target=st.selectbox('Целевая переменная',list(frame.columns),index=list(frame.columns).index('price') if 'price' in frame else len(frame.columns)-1,key=f'target_{signature}')
    available=[c for c in frame if c!=target]
    features=st.multiselect('Признаки',available,default=available,key=f'features_{signature}_{target}')
    st.caption('Остальные столбцы будут исключены.')
    kinds={}
    with st.expander('Типы признаков'):
        for name in features:
            selected=st.selectbox(name,['auto','numeric','categorical','datetime'],key=f'type_{signature}_{name}')
            if selected!='auto':
                kinds[name]=selected
    temporal=None
    if st.checkbox('Временная задача с заранее подготовленной целью',key='train_temporal'):
        st.info('Укажите дату момента прогноза и цель, уже подготовленную для горизонта. Лаги здесь автоматически не создаются.')
        time=st.selectbox('Дата момента прогноза',list(frame.columns),key=f'time_{signature}')
        frequency=st.selectbox('Частота',['MS','D','W-MON','h'],key='train_frequency')
        horizon=st.number_input('Горизонт (периодов)',min_value=1,value=1,key='train_horizon')
        definition=st.text_input('Что означает подготовленная цель',key='train_definition')
        confirmed=st.checkbox('Все выбранные признаки известны на момент прогноза',key='train_available')
        temporal=dict(time_column=time,frequency=frequency,horizon=int(horizon),target_definition=definition,
                      available_features=features if confirmed else [])
    options,hidden=training_options('train')
    kind=st.selectbox('Модель для основного результата',['mlp','linear','baseline','best'])
    if st.button('Обучить и сравнить',type='primary',disabled=not features):
        from predictml.tabular.application import train_file
        config=dict(target=target,features=features,exclude=[c for c in available if c not in features],types=kinds,seed=options['seed'],temporal=temporal)
        result=guarded(lambda:train_file(path,store.root,config,sheet=sheet,settings=TrainingConfig(**options),hidden_sizes=layers(hidden),model_kind=kind))
        if result is not None:
            st.session_state['train_result']=(str(store.root),str(store.root/'runs'/result.storage['run_id']))
            st.success('Обучение завершено. Версии сохранены; доступны в разделе «Версии и прогноз».')
    result=st.session_state.get('train_result')
    if result and result[0]==str(store.root):
        st.caption('Результат последнего выполненного обучения. Изменение формы не переобучает модель.')
        artifacts(result[1])


def scenario_page(store,workspace):
    st.header('Спрос, закупки и рекомендация цены')
    task=st.selectbox('Сценарий',['materials','price'],format_func=lambda x:'Спрос и закупки' if x=='materials' else 'Рекомендация цены')
    if task=='price':
        st.info('Модель воспроизводит формулу рекомендации цены. Это не прогноз будущей рыночной цены.')
    with st.expander('Подготовить исходный экспорт ServiceFlow',expanded=False):
        st.caption('Нужны notes.csv, services.csv, materials.csv (с unit/purchase_step), service_materials.csv. После экспорта Rails и БД не используются.')
        raw=data_directory('raw',workspace)
        cutoff=st.date_input('Граница данных: месяц этой даты исключается')
        if st.button('Подготовить полные месяцы',disabled=raw is None):
            from predictml.preparation import prepare
            import uuid
            destination=Path(workspace)/'prepared'/str(uuid.uuid4())
            result=guarded(lambda:(prepare(raw,destination,str(cutoff)),str(destination))[1])
            if result:
                st.session_state['prepared_result']=result
                st.success('Подготовка завершена')
        if st.session_state.get('prepared_result'):
            st.code(st.session_state['prepared_result'])
    use_prepared=st.checkbox('Использовать только что подготовленные данные',value=bool(st.session_state.get('prepared_result')))
    data=Path(st.session_state['prepared_result']) if use_prepared and st.session_state.get('prepared_result') else data_directory('scenario',workspace)
    st.caption('Спрос: monthly_demand.csv, preparation.json и два JSON-справочника. Цена: price_analysis.csv.')
    months=st.slider('Горизонт закупки, месяцев',1,3,3) if task=='materials' else 3
    options,hidden=training_options('scenario')
    if st.button('Обучить модели сценария',type='primary',disabled=data is None):
        from predictml.comparison import compare_service
        result=guarded(lambda:compare_service(task,data,store.root,settings=TrainingConfig(**options),hidden_sizes=layers(hidden),months=months))
        if result:
            st.session_state['scenario_result']=(str(store.root),task,str(data),months,str(store.root/'runs'/result['run_id']))
            st.success('MLP и линейная модель сохранены')
    result=st.session_state.get('scenario_result')
    if result and result[:4]==(str(store.root),task,str(data),months):
        artifacts(result[4])


def version_select(store,key,multiple=False):
    records=store.versions()
    mapping={r['id']:r for r in records}
    if not records:
        st.info('Сохранённых версий пока нет. Сначала обучите модель.')
        return [] if multiple else None
    if multiple:
        return st.multiselect('Версии',list(mapping),format_func=lambda v:version_label(mapping[v]),key=key)
    return st.selectbox('Версия',list(mapping),format_func=lambda v:version_label(mapping[v]),key=key)


def versions_page(store,workspace):
    st.header('Версии и прогноз')
    version=version_select(store,'prediction_version')
    if not version:
        return
    result=guarded(lambda:store.version(version))
    if result is None:
        return
    record,bundle,manifest=result
    st.caption(f"ID: {version} · родитель: {record['parent_id'] or 'корневая модель'}")
    with st.expander('Схема, параметры и метрики'):
        st.json(manifest)
    task=manifest['task']
    sheet=None
    if task=='tabular':
        path,sheet,frame=table_input('predict',workspace)
    else:
        path=data_directory('predict_domain',workspace)
        st.caption('Для спроса достаточно materials.json и service_materials.json: история сохранена в модели. Для цены нужен price_analysis.csv.')
    months=st.slider('Месяцев прогноза',1,3,3,key='predict_months') if task=='materials' else 3
    if st.button('Получить прогноз',type='primary',disabled=path is None):
        from predictml.storage.operations import predict_version
        result=guarded(lambda:predict_version(store.root,version,path,months=months,sheet=sheet))
        if result:
            st.session_state['prediction_result']=(str(store.root),version,str(path),sheet,months,result['output_path'])
            st.success('Прогноз готов, обучение не запускалось')
    result=st.session_state.get('prediction_result')
    if result and result[:5]==(str(store.root),version,str(path),sheet,months):
        artifacts(result[5])
    if st.button('Экспортировать метрики версии'):
        from predictml.storage.operations import export_metrics
        import uuid
        export_dir=Path(workspace)/'exports'/str(uuid.uuid4())
        result=guarded(lambda:[str(export_metrics(store.root,version,export_dir/f'metrics.{ext}')) for ext in ('csv','xlsx')])
        if result:
            st.session_state['metrics_result']=(version,str(export_dir))
    result=st.session_state.get('metrics_result')
    if result and result[0]==version:
        artifacts(result[1])


def updates_page(store,workspace):
    st.header('Обновление модели')
    version=version_select(store,'update_version')
    if not version:
        return
    loaded=guarded(lambda:store.version(version))
    if loaded is None:
        return
    manifest=loaded[2]
    operations=['retrain','fine-tune'] if manifest['model_kind']=='mlp' else ['retrain']
    operation=st.radio('Операция',operations,format_func=lambda x:'С нуля' if x=='retrain' else 'Продолжить MLP с фиксированным предобработчиком')
    st.caption('Создаётся новая версия. При продолжении используется новый Adam; предобработчик и архитектура сохраняются.')
    sheet=None
    if manifest['task']=='tabular':
        path,sheet,frame=table_input('update_data',workspace)
    else:
        path=data_directory('update_data_domain',workspace)
    config_upload=st.file_uploader('Новая конфигурация таблицы JSON (необязательно, только с нуля)',type=['json'],disabled=operation!='retrain' or manifest['task']!='tabular')
    options,hidden=training_options('update',fixed_layers=manifest['architecture']['hidden_sizes'] if operation=='fine-tune' else None)
    if st.button('Создать обновлённую версию',type='primary',disabled=path is None):
        from predictml.updates import update_version
        def execute():
            config=json.loads(config_upload.getvalue()) if config_upload and operation=='retrain' else None
            return update_version(store.root,version,path,operation=operation,settings=TrainingConfig(**options),config=config,
                                  hidden_sizes=layers(hidden) if operation=='retrain' else None,sheet=sheet)
        result=guarded(execute)
        if result:
            st.session_state['update_result']=(str(store.root),version,str(path),operation,result)
    result=st.session_state.get('update_result')
    if result and result[:4]==(str(store.root),version,str(path),operation):
        st.success('Новая версия сохранена')
        st.json(result[4])


def control_page(store,workspace):
    st.header('Сравнение версий на общем контроле')
    st.info('Контроль должен быть заранее отложен и не участвовать в обучении или подборе настроек. Пересечения и непроверенное происхождение отклоняются.')
    versions=version_select(store,'control_versions',multiple=True)
    path,sheet,frame=table_input('control',workspace)
    history=st.file_uploader('Общая история спроса CSV (необязательно)',type=['csv'])
    import hashlib
    history_signature=hashlib.sha256(history.getvalue()).hexdigest() if history else None
    if st.button('Сравнить без обучения',type='primary',disabled=len(versions)<2 or path is None):
        from predictml.version_comparison import compare_versions
        result=guarded(lambda:compare_versions(store.root,versions,path,sheet=sheet,history=save_upload(history,workspace) if history else None))
        if result:
            st.session_state['control_result']=(str(store.root),tuple(versions),str(path),sheet,history_signature,result['output_path'])
    result=st.session_state.get('control_result')
    if result and result[:5]==(str(store.root),tuple(versions),str(path),sheet,history_signature):
        artifacts(result[5])


def history_page(store):
    st.header('История запусков')
    records=store.runs()
    st.dataframe(pd.DataFrame(records),hide_index=True)
    if not records:
        return
    selected=st.selectbox('Запуск',[r['id'] for r in records],key='history_run')
    record=next(r for r in records if r['id']==selected)
    with st.expander('Параметры и статус',expanded=True):
        st.json(record)
    if record['output_path']:
        artifacts(store.root/record['output_path'])
    elif record['error']:
        st.error(record['error'])
    if record['status']=='running':
        stopped=st.checkbox('Выполнение этого запуска уже остановлено',key='stopped_'+selected)
        if st.button('Отметить запуск прерванным',disabled=not stopped):
            result=guarded(lambda:(store.recover(selected),True)[1])
            if result: st.rerun()


def main():
    st.set_page_config(page_title='PredictML',page_icon='📈',layout='wide')
    st.title('PredictML')
    st.caption('Локальная регрессия: данные → обучение → сравнение → прогноз → обновление')
    workspace=st.sidebar.text_input('Рабочая папка',value=os.environ.get('PREDICTML_WORKSPACE',str(Path.cwd()/'artifacts/ui')))
    store_path=st.sidebar.text_input('Хранилище моделей',value=os.environ.get('PREDICTML_STORE',str(Path.cwd()/'artifacts/ui/models')))
    page=st.sidebar.radio('Раздел',['Таблица','Спрос и цена','Версии и прогноз','Обновление','Контроль','История'])
    store=guarded(lambda:Store(Path(store_path).expanduser()))
    if store is None:
        return
    try:
        if page=='Таблица': train_page(store,workspace)
        elif page=='Спрос и цена': scenario_page(store,workspace)
        elif page=='Версии и прогноз': versions_page(store,workspace)
        elif page=='Обновление': updates_page(store,workspace)
        elif page=='Контроль': control_page(store,workspace)
        else: history_page(store)
    except Exception as exc:
        st.error(f'Проверьте выбранные данные и параметры: {exc}')
        with st.expander('Подробности'):
            st.code(traceback.format_exc())
    st.sidebar.caption(f'Сохранено версий: {len(store.versions())}')


if __name__=='__main__':
    main()
