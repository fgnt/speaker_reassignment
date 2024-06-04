#!/usr/bin/env python

from pathlib import Path
import re
import json
from speaker_reassignment.utils import download, run


def maybe_download(file):
    if Path(file).exists():
        print(f'Skipping download of {file}. Already exists.')
    else:
        url = f'https://huggingface.co/datasets/boeddeker/libri_css_tssep_gss_wavLMASR/resolve/main/{file}?download=true'
        download(url, file)


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
