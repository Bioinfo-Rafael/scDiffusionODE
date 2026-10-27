"""Result guide: existing-file links, safe labels, incomplete runs, no scientific imports."""
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from urllib.parse import unquote

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from report import write_report
from figure_guide import describe_figure


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and not value.startswith('#'):
                self.paths.append(unquote(value))


class ReportTests(unittest.TestCase):
    def test_partial_result_has_only_real_links_and_does_not_touch_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'analysis'; root.mkdir()
            (root / 'manifest.json').write_text(json.dumps({'status': 'failed', 'mode': 'smoke'}))
            folder = root / 'matsum/t049/coefficients/groups/00_<test>'
            folder.mkdir(parents=True)
            image = folder / 'signed_histograms.png'; image.write_bytes(b'unchanged fixture')
            before = image.read_bytes()
            page = write_report(root.parent).read_text()
            self.assertIn('人工データ', page)
            self.assertIn('完了を確認できていません', page)
            self.assertNotIn('00_<test>', page)
            self.assertEqual(before, image.read_bytes())
            parser = Links(); parser.feed(page)
            self.assertTrue(parser.paths)
            for value in parser.paths:
                self.assertTrue((root / value).is_file(), value)

    def test_coefficient_and_basis_variances_are_distinguished(self):
        basis = ' '.join(describe_figure(Path('basis/across_basis_std_histogram.png')))
        cells = ' '.join(describe_figure(Path('cell_matrices/groups/all/W_std_histogram.png')))
        self.assertIn('8個のB_k方向', basis)
        self.assertIn('細胞方向', cells)

    def test_cli_needs_no_numerical_or_plotting_libraries(self):
        script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy','torch','matplotlib','sklearn','umap','pandas'):
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Block())
import report
report.main()
'''
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'matsum').mkdir()
            result = subprocess.run([sys.executable, '-c', script, tmp], cwd=HERE, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(tmp) / 'index.html').is_file())


if __name__ == '__main__': unittest.main()
