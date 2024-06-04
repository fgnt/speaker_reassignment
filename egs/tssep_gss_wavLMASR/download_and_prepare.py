#!/usr/bin/env python

import subprocess
import shlex
from pathlib import Path
import re
import json


class c:
    green = '\033[92m'
    end = '\033[0m'


def run(cmd):
    cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f'{c.green}$ {cmd_str}{c.end}')
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True)


def maybe_download(file):
    if Path(file).exists():
        print(f'Skipping download of {file}. Already exists.')
    else:
        run(f'wget https://huggingface.co/boeddeker/libri_css_tssep_gss_wavLMASR/resolve/main/{file}')
        # run(f'cp huggingface/{file} .')


def main():
    if not Path('audio').exists():
        maybe_download('audio.zip')
        run(f'unzip audio.zip')
    else:
        print(f'Skipping extraction of audio. Already exists.')

    maybe_download('hyp_orig.json')
    maybe_download('ref.stm')

    hyp = Path('hyp_orig.json').read_text()
    # Fix the audio_path. Don't use json to avoid changes to the timestamps.
    new = re.sub(
        '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/77/eval/62000/71/audio_gss_mask/([^/]+)/([^/]+).wav',
        f'{Path.cwd() / "audio"}/\\2.flac',
        hyp,
    )
    Path('hyp.json').write_text(new)
    print(f'Wrote hyp.json with fixed paths.')
    assert Path(json.loads(new)[0]['audio_path']).exists(), (json.loads(new)[0]['audio_path'], 'Something went wrong. The file does not exist.')


if __name__ == '__main__':
    main()
