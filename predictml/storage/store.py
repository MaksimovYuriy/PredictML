"""One atomic run directory, then one SQLite transaction publishing versions."""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
import warnings
from .bundles import write_bundle, digest, verify_bundle, load_payload


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=lambda value: asdict(value) if is_dataclass(value) else str(value))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        for ancestor in (self.root, *self.root.parents):
            if (ancestor/'manifest.json').is_file() and ((ancestor/'model.joblib').is_file() or (ancestor/'model.pth').is_file()):
                raise ValueError('cannot create a store inside a model bundle')
            if ancestor != self.root and (ancestor/'history.sqlite3').exists() and self.root.is_relative_to(ancestor/'runs'):
                raise ValueError('cannot create a store inside immutable run history')
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root/'history.sqlite3'
        with self.connect() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('unsupported SQLite history schema')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, operation TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT, parameters TEXT NOT NULL,
                    output_path TEXT, input_version TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    created_at TEXT NOT NULL, task TEXT NOT NULL, model_kind TEXT NOT NULL,
                    target TEXT NOT NULL, schema_json TEXT NOT NULL, bundle_path TEXT NOT NULL UNIQUE,
                    manifest_hash TEXT NOT NULL, parent_id TEXT REFERENCES versions(id));
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def check_destination(self, path):
        if Path(path).resolve().is_relative_to(self.root/'runs'):
            raise ValueError('export destination cannot overwrite immutable run history')

    def start(self, operation, parameters, input_version=None):
        run_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute('INSERT INTO runs(id,operation,status,started_at,parameters,input_version) VALUES(?,?,?,?,?,?)',
                       (run_id, operation, 'running', now(), encode(parameters), input_version))
        return run_id

    def fail(self, run_id, error):
        with self.connect() as db:
            db.execute("UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=? AND status='running'",
                       (now(), str(error), run_id))

    def complete(self, run_id, directory, manifests):
        with self.connect() as db:
            status = db.execute('SELECT status FROM runs WHERE id=?', (run_id,)).fetchone()
            if status is None or status['status'] != 'running':
                raise ValueError('run is not running')
            for manifest in manifests:
                bundle = directory/'versions'/manifest['version_id']
                db.execute('INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (manifest['version_id'], run_id, manifest['created_at'], manifest['task'],
                            manifest['model_kind'], manifest['schema']['target'], encode(manifest['schema']),
                            str(bundle.relative_to(self.root)), digest(bundle/'manifest.json'), manifest.get('parent_id')))
            db.execute("UPDATE runs SET status='completed',finished_at=?,output_path=? WHERE id=?",
                       (now(), str(directory.relative_to(self.root)), run_id))

    def runs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM runs ORDER BY started_at,id')]

    def recover(self, run_id):
        """Explicitly mark an interrupted run failed; never guess if it is alive."""
        with self.connect() as db:
            changed = db.execute("UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=? AND status='running'",
                                 (now(), 'Manually marked interrupted; unregistered files retained for inspection', run_id)).rowcount
            if not changed:
                raise ValueError('only a running run can be marked interrupted')

    def versions(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM versions ORDER BY created_at,id')]

    def version(self, version_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM versions WHERE id=?', (version_id,)).fetchone()
        if row is None:
            raise ValueError(f'unknown model version: {version_id}')
        record = dict(row)
        path = (self.root/record['bundle_path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('version path escapes store')
        manifest = verify_bundle(path, manifest_hash=record['manifest_hash'])
        if manifest['version_id'] != version_id or manifest['run_id'] != record['run_id']:
            raise ValueError('version identity differs from history')
        return record, path, manifest

    def load(self, version_id):
        record, path, _ = self.version(version_id)
        manifest, payload = load_payload(path, manifest_hash=record['manifest_hash'])
        return path, manifest, payload


def execute_run(root, operation, parameters, worker, specs=None, *, input_version=None, publish=True):
    """worker(stage, run_id); specs(stage, result) -> bundle metadata + source.

    Compatibility aliases at root are replaceable; runs/<id>/ is authoritative.
    """
    store = Store(root)
    run_id = store.start(operation, parameters, input_version)
    runs = store.root/'runs'
    stage, final = runs/f'.pending-{run_id}', runs/run_id
    manifests = []
    try:
        runs.mkdir(exist_ok=True)
        stage.mkdir()
        result = worker(stage, run_id)
        for spec in specs(stage, result) if specs else []:
            spec = dict(spec)
            source = spec.pop('source')
            version_id = str(uuid.uuid4())
            manifest = write_bundle(stage/'versions'/version_id, source,
                                    {**spec, 'version_id': version_id, 'run_id': run_id, 'created_at': now()})
            manifests.append(manifest)
        (stage/'versions.json').write_text(encode([dict(version_id=m['version_id'], task=m['task'], model_kind=m['model_kind']) for m in manifests]), encoding='utf-8')
        os.replace(stage, final)
        store.complete(run_id, final, manifests)
    except BaseException as exc:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
        store.fail(run_id, f'{type(exc).__name__}: {exc}')
        raise
    if publish:
        # These files only preserve old command paths; versions never use aliases.
        try:
            for source in final.iterdir():
                if source.is_file():
                    temporary = store.root/f'.alias-{run_id}-{source.name}'
                    try:
                        shutil.copyfile(source, temporary)
                        os.replace(temporary, store.root/source.name)
                    finally:
                        temporary.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(f'run {run_id} is committed at {final}; latest-file aliases could not be refreshed: {exc}', RuntimeWarning)
    return result, run_id, [m['version_id'] for m in manifests]
