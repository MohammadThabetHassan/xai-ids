"""Build the Kaggle notebook that runs revision/xai_ids_revised.py four times
(seed 42 with the sensitivity analysis, the chronological CSE-CIC-IDS-2018 split,
seed 7 and seed 2024). This is the notebook behind version 3 of the results.

    py revision/build_kaggle_notebook.py revision/xai_ids_revised.py <commit> xai_ids_runs.ipynb

Upload the .ipynb to Kaggle, attach the three datasets (RERUN_PLAN.md), and run it
with "Save & Run All". Outputs go to /kaggle/working/xai_ids_v3/<run>/ and are zipped
to /kaggle/working/xai_ids_v3_results.zip (models removed)."""
import json
import sys
from pathlib import Path

script, commit, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


driver = '''import os, subprocess, sys, time, shutil, json
ROOT = '/kaggle/working/xai_ids_v3'
RUNS = [
    ('seed42',       {'XAI_IDS_SEED': '42', 'XAI_IDS_SENS': '1'}),
    ('temporal2018', {'XAI_IDS_SEED': '42', 'XAI_IDS_SPLIT': 'temporal'}),
    ('seed7',        {'XAI_IDS_SEED': '7'}),
    ('seed2024',     {'XAI_IDS_SEED': '2024'}),
]
status = {}
for name, extra in RUNS:
    out_dir = os.path.join(ROOT, name)
    env = dict(os.environ, XAI_IDS_OUT=out_dir, PYTHONUNBUFFERED='1', **extra)
    print('=' * 20, 'RUN', name, extra, '=' * 20, flush=True)
    t0 = time.time()
    try:
        proc = subprocess.Popen([sys.executable, SCRIPT_PATH], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors='replace')
        with open(os.path.join(ROOT, name + '_log.txt'), 'w') as log:
            for line in proc.stdout:
                log.write(line)
                if 'Debugger' not in line and 'frozen' not in line:
                    print(line, end='', flush=True)
        rc = proc.wait()
    except Exception as e:
        rc = repr(e)
    shutil.rmtree(os.path.join(out_dir, 'models'), ignore_errors=True)
    status[name] = {'returncode': rc, 'minutes': round((time.time() - t0) / 60, 1), 'env': extra}
    print('=' * 20, 'END', name, status[name], '=' * 20, flush=True)
    json.dump(status, open(os.path.join(ROOT, 'status.json'), 'w'), indent=1)
print(status)
print(shutil.make_archive('/kaggle/working/xai_ids_v3_results', 'zip', ROOT))
'''

cells = [
    md(f"# XAI-IDS: seeds, temporal split, baselines and sensitivity\n\n"
       f"Runs `revision/xai_ids_revised.py` at commit `{commit}` of "
       f"https://github.com/MohammadThabetHassan/xai-ids four times.\n"),
    code("import importlib.util, subprocess, sys\n"
         "missing = [pkg for pkg, mod in [('lime', 'lime'), ('imbalanced-learn', 'imblearn')]\n"
         "           if importlib.util.find_spec(mod) is None]\n"
         "print('missing before install:', missing)\n"
         "if missing:\n"
         "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)\n"),
    code("import os\n"
         "os.makedirs('/kaggle/working/xai_ids_v3', exist_ok=True)\n"
         "SCRIPT_PATH = '/kaggle/working/xai_ids_revised.py'\n"
         "SCRIPT = " + json.dumps(script.read_text(encoding="utf-8"), ensure_ascii=False) + "\n"
         "open(SCRIPT_PATH, 'w', encoding='utf-8').write(SCRIPT)\n"
         "print('script written:', len(SCRIPT), 'chars')\n"),
    code(driver),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "|", len(cells), "cells")
