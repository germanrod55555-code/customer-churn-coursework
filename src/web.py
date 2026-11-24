"""Локальная демонстрация: CSV/Parquet → сохранённая модель → таблица риска."""
from __future__ import annotations

import argparse
import io
import json
import tempfile
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
from churn import FEATURES, ROOT, predict_file

MAX_UPLOAD = 32 * 1024 * 1024


def analyze_file(content: bytes, suffix: str, model_path: Path) -> tuple[dict, bytes]:
    """Использует тот же пакетный прогноз, что и командный интерфейс."""
    if suffix not in {'.csv', '.parquet'}:
        raise ValueError('Выберите CSV или Parquet.')
    if not content:
        raise ValueError('Файл пуст. Выберите файл с заголовком и данными.')
    if len(content) > MAX_UPLOAD:
        raise ValueError('Для демонстрации допускается файл до 32 МБ.')
    with tempfile.TemporaryDirectory(prefix='churn_demo_') as folder:
        source = Path(folder) / ('input' + suffix)
        output = Path(folder) / 'prediction.csv'
        source.write_bytes(content)
        count = predict_file(source, output, model_path, batch_size=10000)
        csv_bytes = output.read_bytes()
        frame = pd.read_csv(io.BytesIO(csv_bytes))
    return {
        'count': count,
        'flagged': int(frame['risk_flag'].sum()),
        'preview': frame.head(100).to_dict(orient='records'),
        'preview_limit': 100,
    }, csv_bytes


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address, model_path=None):
        super().__init__(address, Handler)
        self.model_path = model_path or ROOT / 'results/model.joblib'
        self.downloads: dict[str, bytes] = {}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, kind='application/json; charset=utf-8', download=False):
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="prediction.csv"')
        self.end_headers()
        self.wfile.write(content)

    def local_request(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        return host in allowed and (not origin or origin in {'http://' + h for h in allowed})

    def do_GET(self):
        if not self.local_request():
            return self.send(403, {'error': 'Доступ разрешён только с этого компьютера.'})
        path = urlparse(self.path).path
        if path == '/':
            return self.send(200, Path(__file__).with_name('interface.html').read_bytes(), 'text/html; charset=utf-8')
        if path == '/api/example':
            return self.send(200, (ROOT / 'data/example_customers.csv').read_bytes(), 'text/csv; charset=utf-8')
        if path.startswith('/download/'):
            data = self.server.downloads.get(path.rsplit('/', 1)[-1])
            if data is not None:
                return self.send(200, data, 'text/csv; charset=utf-8', True)
        self.send(404, {'error': 'Страница не найдена.'})

    def do_POST(self):
        if not self.local_request():
            return self.send(403, {'error': 'Недопустимый источник запроса.'})
        request = urlparse(self.path)
        if request.path != '/api/predict':
            return self.send(404, {'error': 'Страница не найдена.'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= MAX_UPLOAD:
                raise ValueError('Выберите непустой файл размером до 32 МБ.')
            filename = parse_qs(request.query).get('name', [''])[0]
            result, csv_bytes = analyze_file(self.rfile.read(size), Path(filename).suffix.lower(), self.server.model_path)
            key = uuid.uuid4().hex
            # Не более пяти результатов на время одной демонстрации.
            while len(self.server.downloads) >= 5:
                self.server.downloads.pop(next(iter(self.server.downloads)))
            self.server.downloads[key] = csv_bytes
            result['download'] = '/download/' + key
            self.send(200, result)
        except (ValueError, KeyError, UnicodeError, OSError) as exc:
            self.send(400, {'error': str(exc)})
        except Exception:
            self.send(400, {'error': 'Не удалось прочитать файл. Проверьте формат, признаки и числовые значения.'})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description='Демонстрация прогнозирования оттока')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = DemoServer(('127.0.0.1', args.port))
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Демонстрация: {url}\nДля завершения нажмите Ctrl+C.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
