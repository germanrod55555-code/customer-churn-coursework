"""Проверки нового интерфейса, включая совпадение с командным прогнозом."""
import io
import json
import sys
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from web import DemoServer, analyze_file


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ROOT / 'results/model.joblib'
        cls.example = (ROOT / 'data/example_customers.csv').read_bytes()
        cls.server = DemoServer(('127.0.0.1', 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_same_as_saved_cli_example(self):
        result, raw = analyze_file(self.example, '.csv', self.model)
        actual = pd.read_csv(io.BytesIO(raw))
        expected = pd.read_csv(ROOT / 'results/example_predictions.csv')
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual((result['count'], result['flagged']), (10, 3))

    def test_parquet_same_as_csv(self):
        buffer = io.BytesIO()
        pd.read_csv(io.BytesIO(self.example)).to_parquet(buffer, index=False)
        a, _ = analyze_file(self.example, '.csv', self.model)
        b, _ = analyze_file(buffer.getvalue(), '.parquet', self.model)
        self.assertEqual(a, b)

    def test_header_only_is_valid(self):
        raw = self.example.splitlines()[0] + b'\n'
        result, _ = analyze_file(raw, '.csv', self.model)
        self.assertEqual((result['count'], result['flagged'], result['preview']), (0, 0, []))

    def test_bad_files_are_rejected(self):
        for raw, suffix in [(b'', '.csv'), (self.example, '.txt'), (b'wrong\n1\n', '.csv')]:
            with self.subTest(suffix=suffix, length=len(raw)), self.assertRaises(ValueError):
                analyze_file(raw, suffix, self.model)

    def test_http_prediction_and_download(self):
        request = urllib.request.Request(self.url + '/api/predict?name=sample.csv', data=self.example, method='POST')
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result['flagged'], 3)
        with urllib.request.urlopen(self.url + result['download']) as response:
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertEqual(len(pd.read_csv(response)), 10)

    def test_foreign_origin_is_rejected(self):
        request = urllib.request.Request(self.url + '/api/predict?name=x.csv', data=self.example,
                                         headers={'Origin': 'https://example.org'}, method='POST')
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
