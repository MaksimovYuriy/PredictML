"""Optional real Chromium smoke test; start the UI server separately."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8501')
    parser.add_argument('--examples',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    examples,output=Path(args.examples).resolve(),Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    errors=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        context=browser.new_context(viewport=dict(width=1440,height=1000),accept_downloads=True)
        page=context.new_page();page.set_default_timeout(60000)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url)
        page.get_by_role('heading',name='PredictML',exact=True).wait_for()
        page.locator('input[type=file]').first.set_input_files(examples/'tabular/housing.xlsx')
        page.get_by_label('Лист',exact=True).click()
        page.get_by_role('option',name='training',exact=True).click()
        page.get_by_text('Настройки обучения',exact=True).click()
        epochs=page.get_by_role('spinbutton',name='Максимум эпох',exact=True)
        epochs.fill('3');epochs.press('Enter')
        page.get_by_role('button',name='Обучить и сравнить',exact=True).click()
        page.get_by_text('Обучение завершено. Версии сохранены; доступны в разделе «Версии и прогноз».',exact=True).wait_for()
        page.get_by_text('Метрики',exact=True).scroll_into_view_if_needed()
        page.screenshot(path=output/'training.png',full_page=True)
        with page.expect_download() as event:
            page.get_by_role('button',name='comparison.csv',exact=True).click()
        event.value.save_as(output/'comparison-download.csv')
        page.get_by_text('Версии и прогноз',exact=True).click()
        page.get_by_role('heading',name='Версии и прогноз',exact=True).wait_for()
        page.locator('input[type=file]').first.set_input_files(examples/'tabular/housing_new.csv')
        page.get_by_role('button',name='Получить прогноз',exact=True).click()
        page.get_by_text('Прогноз готов, обучение не запускалось',exact=True).wait_for()
        page.screenshot(path=output/'prediction.png',full_page=True)
        with page.expect_download() as event:
            page.get_by_role('button',name='results.xlsx',exact=True).click()
        event.value.save_as(output/'prediction-download.xlsx')
        browser.close()
    if errors:
        raise RuntimeError('; '.join(errors))
    (output/'browser.json').write_text(json.dumps(dict(status='passed',javascript_errors=errors,
        checks=['real XLSX upload','sheet selection','MLP/linear/baseline training','CSV download',
                'version selection','new CSV upload','prediction','XLSX download']),indent=2),encoding='utf-8')


if __name__=='__main__':main()
