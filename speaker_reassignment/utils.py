import sys
import time
import urllib.request
import subprocess
import shlex
from pathlib import Path

class Reporthook:
    start_time = 0

    def __call__(self, count, block_size, total_size):
        # https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
        if count == 0:
            self.start_time = time.time()
            return
        duration = time.time() - self.start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
            (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()


def download(url, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {file} from {url}.')
    urllib.request.urlretrieve(url, file, Reporthook())
    print(f'Downloaded {file}.')


class c:
    green = '\033[92m'
    end = '\033[0m'


def run(cmd):
    cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f'{c.green}$ {cmd_str}{c.end}')
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True)
