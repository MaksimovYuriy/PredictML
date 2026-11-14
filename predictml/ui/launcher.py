"""Local browser UI launcher; core usage does not require importing Streamlit."""
import os
from pathlib import Path
import sys


def launch(workspace=None, store=None, port=8501):
    if not 1 <= port <= 65535:
        raise ValueError('port must be in [1,65535]')
    if workspace:
        os.environ['PREDICTML_WORKSPACE']=str(Path(workspace).expanduser().resolve())
    if store:
        os.environ['PREDICTML_STORE']=str(Path(store).expanduser().resolve())
    from streamlit.web import cli
    sys.argv=['streamlit','run',str(Path(__file__).with_name('app.py')),
              '--global.developmentMode=false','--server.address=127.0.0.1','--server.headless=true',f'--server.port={port}','--browser.gatherUsageStats=false']
    cli.main()
