import sys

import click


def main(*args):
    lines = {}

    for a in args[0][1:]:
        with open(a, 'r') as f:
            for i, l in enumerate(f):
                if l.startswith("==="):
                    lines[i] = l
                elif l.startswith("scenario"):
                    lines[i] = l
                else:
                    lines.setdefault(i, [])
                    lines[i].append(l.split(','))

    for k in range(len(lines)):
        if isinstance(lines[k], list):
            print(
                ','.join(
                    map(
                        lambda x: str(sum(map(float, x)) / len(x)),
                        zip(*lines[k])
                    )
                ),
            )
        else:
            print(lines[k], end='')


if __name__ == '__main__':
    main(sys.argv)
