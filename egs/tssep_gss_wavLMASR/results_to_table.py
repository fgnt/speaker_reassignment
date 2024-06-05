#!/usr/bin/env python
from pathlib import Path
import paderbox as pb


def print_table(table: 'list[dict]', filler='N/A', align='<>'):
    """
    >>> table = [{'abc': 1, 'b': 222}, {'a': 3, 'c': 4}]
    >>> print_table(table)
    abc | b   | a   | c
    --- + --- + --- + ---
    1   | 222 | N/A | N/A
    N/A | N/A |   3 |   4

    """
    keys = list(dict.fromkeys(
        key
        for d in table
        for key in d.keys()
    ).keys())

    width = [
        max(len(key), max(len(str(d.get(key, filler))) for d in table))
        for key in keys
    ]

    sep = ' | '

    cells = []
    for k, w in zip(keys, width):
        cells.append(f'{k:<{w}}')
    print(*cells, sep=sep)
    print(*['-'*len(c) for c in cells], sep=sep.replace('|', '+'))

    # repeat the last character of align to the length of keys
    align = align.ljust(len(keys), align[-1])

    for d in table:
        cells = []
        for i, (k, w) in enumerate(zip(keys, width)):
            cells.append(f'{str(d.get(k, filler)):{align[i]}{w}}')
        print(*cells, sep=sep)


def main():

    table = []

    files = list(Path('.').glob('hyp*.json'))

    # sort by timestamp
    files.sort(key=lambda x: x.stat().st_mtime)

    for file in files:
        data = pb.io.load(file)
        if isinstance(data, dict) and 'error_rate' in data:
            data.pop('assignment', None)
            data.pop('hypothesis_self_overlap', None)
            data.pop('reference_self_overlap', None)
            data['error_rate'] = f'{data["error_rate"]:.2%}'
            data = {
                'file': file.name,
                **{
                    k: f'{v:_}' if isinstance(v, int) else v
                    for k, v in data.items()
                },
            }
            table.append(data)

    print_table(table)


if __name__ == '__main__':
    main()

