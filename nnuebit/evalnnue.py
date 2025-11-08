#!/usr/bin/env python3

import argparse
import subprocess
import sys

def tcadjust(tc: str) -> str:
    f = open('/etc/bitbit/tcfactor', 'r')
    tcfactor = float(f.read().strip())
    f.close()

    moves: int = 0
    maintime: float = 0
    increment: float = 0

    i = tc.find('/')
    if i != -1:
        moves = int(tc[:i])
        tc = tc[i + 1:]
    i = tc.find('+')
    if i != -1:
        maintime = float(tc[:i])
        increment = float(tc[i + 1:])
    else:
        maintime = float(tc)

    tc = ''
    if moves > 0:
        tc += f'{moves}/'
    tc += f'{tcfactor * maintime}'
    if increment > 0:
        tc += f'+{tcfactor * increment}'

    return tc

def numcpus(cpus: str) -> int:
    if cpus == '' or cpus.startswith(',') or cpus.endswith(',') or cpus.startswith('-') or cpus.endswith('-'):
        raise ValueError

    if ',' in cpus:
        s = cpus.split(',', 1)
        return numcpus(s[0]) + numcpus(s[1])

    if '-' in cpus:
        if cpus.count('-') > 1:
            raise ValueError('Only one \'-\' per \',\'')
        s = cpus.split('-')
        if not s[0].isdigit():
            raise ValueError('\'%s\' is not an integer' % (s[0], ))
        if not s[1].isdigit():
            raise ValueError('\'%s\' is not an integer' % (s[0], ))
        a = int(s[0])
        b = int(s[1])
        if b < a:
            raise ValueError('Invalid expression: \'%d-%d\'' % (a, b))

        return b + 1 - a

    if not cpus.isdigit():
        raise ValueError('\'%s\' is not an integer' % (cpus, ))

    return 1

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bitbit', type=str, help='bitbit binary')
    parser.add_argument('book', type=str, help='Opening book (.epd)')
    parser.add_argument('cpus', type=str, help='Cpus in the format 1,2,4-6,8')
    parser.add_argument('--tc', type=str, help='Time control', default='40/5+0.05')
    parser.add_argument('nnue', type=str, nargs='+', help='NNUE files')
    parser.add_argument('--reference', type=str, help='bitbit reference binary', default=None)

    args = parser.parse_args()

    engines = len(args.nnue)
    if args.reference is not None:
        engines += 1

    if engines < 2:
        print('Input at least two NNUE networks or one NNUE network and a reference.')
        sys.exit(1)

    command = ['fastchess',
               '-use-affinity', args.cpus,
               '-tournament', 'roundrobin',
               '-games', '2',
               '-rounds', '10000',
               '-concurrency', str(numcpus(args.cpus)),
               '-each', 'proto=uci', f'tc={tcadjust(args.tc)}',
               '-openings', 'format=epd', f'file={args.book}', 'order=random',
               '-repeat',
               '-ratinginterval', '2',
               '-pgnout', 'file=nnue.pgn']

    for nnue in args.nnue:
        command.extend(['-engine', f'name={nnue.split('/')[-1]}', f'cmd={args.bitbit}', f'option.FileNNUE={nnue}'])
    if args.reference is not None:
        command.extend(['-engine', f'name=reference', f'cmd={args.reference}'])

    print(*command, sep=' ')
    subprocess.run(command)

if __name__ == '__main__':
    main()
