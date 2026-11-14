from pathlib import Path
import io
import os
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from predictml.ui.helpers import save_upload, save_group, layers
from predictml.tabular.io import read_table
from predictml.storage import Store

ROOT=Path(__file__).resolve().parents[1]


def widget(collection,label):
    return next(w for w in collection if w.label==label)


class Uploaded(io.BytesIO):
    def __init__(self,name,data):
        super().__init__(data);self.name=name


class UITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        env=patch.dict(os.environ,{'PREDICTML_WORKSPACE':str(self.root),'PREDICTML_STORE':str(self.root/'models')})
        env.start();self.addCleanup(env.stop)

    def assert_clean(self,at):
        self.assertFalse(list(at.exception),[e.message for e in at.exception])
        self.assertFalse(list(at.error),[e.value for e in at.error])

    def test_initial_ui_empty_store_and_validation(self):
        at=AppTest.from_file(ROOT/'predictml/ui/app.py',default_timeout=60).run()
        self.assert_clean(at)
        self.assertEqual(widget(at.selectbox,'Целевая переменная').value,'price')
        widget(at.multiselect,'Признаки').set_value([]).run()
        self.assertTrue(widget(at.button,'Обучить и сравнить').disabled)
        widget(at.sidebar.radio,'Раздел').set_value('Версии и прогноз').run()
        self.assert_clean(at)
        self.assertTrue(any('Сохранённых версий пока нет' in i.value for i in at.info))

    def test_full_table_train_predict_update_control_and_restart(self):
        paths={'train':'housing.csv','predict':'housing_new.csv','update_data':'housing_update.csv','control':'housing_control.csv'}
        def tables(key,workspace,**kwargs):
            path=ROOT/'examples/tabular'/paths[key]
            return path,None,read_table(path)
        with patch('predictml.ui.app.table_input',side_effect=tables):
            at=AppTest.from_string('from predictml.ui.app import main\nmain()',default_timeout=60).run()
            widget(at.multiselect,'Признаки').set_value(['area','district','built_at']).run()
            widget(at.selectbox,'built_at').set_value('datetime').run()
            widget(at.number_input,'Максимум эпох').set_value(2).run()
            widget(at.text_input,'Скрытые слои (через запятую)').set_value('4').run()
            widget(at.button,'Обучить и сравнить').click().run()
            self.assert_clean(at)
            store=Store(self.root/'models')
            self.assertEqual(len(store.versions()),3)
            mlp=next(v['id'] for v in store.versions() if v['model_kind']=='mlp')
            widget(at.sidebar.radio,'Раздел').set_value('Версии и прогноз').run()
            widget(at.selectbox,'Версия').set_value(mlp).run()
            widget(at.button,'Получить прогноз').click().run()
            self.assert_clean(at)
            self.assertTrue(any(r['operation']=='predict-version' for r in store.runs()))
            paths['predict']='housing_control.csv'
            at.run()
            self.assertEqual(len(at.dataframe),0, 'previous forecast must be hidden after input changes')
            paths['predict']='housing_new.csv'
            at.run()
            widget(at.button,'Экспортировать метрики версии').click().run()
            self.assert_clean(at)
            widget(at.sidebar.radio,'Раздел').set_value('Обновление').run()
            widget(at.selectbox,'Версия').set_value(mlp).run()
            widget(at.radio,'Операция').set_value('fine-tune').run()
            widget(at.number_input,'Максимум эпох').set_value(2).run()
            widget(at.button,'Создать обновлённую версию').click().run()
            self.assert_clean(at)
            child=next(v['id'] for v in store.versions() if v['parent_id']==mlp)
            widget(at.sidebar.radio,'Раздел').set_value('Контроль').run()
            widget(at.multiselect,'Версии').set_value([mlp,child]).run()
            widget(at.button,'Сравнить без обучения').click().run()
            self.assert_clean(at)
            self.assertTrue(any(r['operation']=='compare-versions' and r['status']=='completed' for r in store.runs()))
            widget(at.sidebar.radio,'Раздел').set_value('История').run()
            self.assert_clean(at)
            restarted=AppTest.from_string('from predictml.ui.app import main\nmain()',default_timeout=60).run()
            widget(restarted.sidebar.radio,'Раздел').set_value('Версии и прогноз').run()
            widget(restarted.selectbox,'Версия').set_value(child).run()
            widget(restarted.button,'Получить прогноз').click().run()
            self.assert_clean(restarted)
            self.assertEqual(len(store.versions()),4)

    def test_domain_scenarios_through_ui(self):
        from predictml.preparation import prepare
        data=self.root/'prepared';prepare(ROOT/'examples/raw',data,'2027-01-01')
        def directory(key,workspace):
            return None if key=='raw' else data
        with patch('predictml.ui.app.data_directory',side_effect=directory):
            at=AppTest.from_string('from predictml.ui.app import main\nmain()',default_timeout=60).run()
            widget(at.sidebar.radio,'Раздел').set_value('Спрос и цена').run()
            for task in ('materials','price'):
                widget(at.selectbox,'Сценарий').set_value(task).run()
                widget(at.number_input,'Максимум эпох').set_value(2).run()
                widget(at.text_input,'Скрытые слои (через запятую)').set_value('4').run()
                widget(at.button,'Обучить модели сценария').click().run()
                self.assert_clean(at)
            store=Store(self.root/'models')
            self.assertEqual(len(store.versions()),4)
            widget(at.sidebar.radio,'Раздел').set_value('Версии и прогноз').run()
            for task in ('materials','price'):
                version=next(v['id'] for v in store.versions() if v['task']==task and v['model_kind']=='linear')
                widget(at.selectbox,'Версия').set_value(version).run()
                widget(at.button,'Получить прогноз').click().run()
                self.assert_clean(at)

    def test_invalid_layers_show_error_and_failed_input_does_not_train(self):
        at=AppTest.from_file(ROOT/'predictml/ui/app.py',default_timeout=60).run()
        widget(at.text_input,'Скрытые слои (через запятую)').set_value('bad').run()
        widget(at.button,'Обучить и сравнить').click().run()
        self.assertFalse(list(at.exception))
        self.assertTrue(any('Скрытые слои' in e.value for e in at.error))
        self.assertEqual(Store(self.root/'models').versions(),[])

    def test_upload_helpers_and_names(self):
        upload=Uploaded('../../input.csv',b'x,y\n1,2\n')
        path=save_upload(upload,self.root)
        self.assertEqual(path.name,'input.csv')
        self.assertEqual(path.read_bytes(),upload.getvalue())
        self.assertEqual(path,save_upload(upload,self.root))
        group=save_group([Uploaded('materials.json',b'[]')],self.root)
        self.assertTrue((group/'materials.json').exists())
        with self.assertRaises(ValueError):
            save_group([Uploaded('a.csv',b''),Uploaded('a.csv',b'')],self.root)
        self.assertEqual(layers('64, 32'),(64,32))
        with self.assertRaises(ValueError): layers('0')

    def test_local_launcher_configuration(self):
        from predictml.ui.launcher import launch
        import sys
        with patch.object(sys,'argv',[]):
            with patch('streamlit.web.cli.main') as main:
                launch(self.root,self.root/'models',18502)
                main.assert_called_once()
                self.assertIn('--server.address=127.0.0.1',sys.argv)
                self.assertIn('--global.developmentMode=false',sys.argv)
                self.assertIn('--browser.gatherUsageStats=false',sys.argv)
        with self.assertRaises(ValueError): launch(port=0)
