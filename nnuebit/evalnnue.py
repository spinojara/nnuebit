#!/usr/bin/env python3

import argparse
import subprocess
import sys

def tcadjust(tc):
    f = open('/etc/bitbit/tcfactor', 'r')
    tcfactor = float(f.read().strip())
    f.close()

    moves = 0
    maintime = 0
    increment = 0

    i = tc.find('/')
    if i != -1:
        moves = int(tc[:i])
        tc = tc[i + 1:]
    i = tc.find('+')
    if i != -1:
        maintime = float(tc[:i])
        increment = float(tc[i + 1:])
    else:
        maintime = tc

    tc = ''
    if moves > 0:
        tc += f'{moves}/'
    tc += f'{tcfactor * maintime}'
    if increment > 0:
        tc += f'+{tcfactor * increment}'

    return tc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bitbit', type=str, help='bitbit binary')
    parser.add_argument('book', type=str, help='Opening book (.epd)')
    parser.add_argument('concurrency', type=int, help='Concurrency')
    parser.add_argument('--tc', type=str, help='Time control', default='40/5+0.05')
    parser.add_argument('nnue', type=str, nargs='+', help='NNUE files')
    parser.add_argument('--reference', type=str, help='bitbit binary reference', default=None)

    args = parser.parse_args()

    engines = len(args.nnue)
    if args.reference is not None:
        engines += 1

    if engines < 2:
        print('Input at least two NNUE networks.')
        sys.exit(1)

    command = ['fastchess',
               '-use-affinity', '2,4,6,8,10',
               '-tournament', 'roundrobin',
               '-games', '2',
               '-rounds', '10000',
               '-concurrency', str(args.concurrency),
               '-each', 'proto=uci', f'tc={tcadjust(args.tc)}',
               '-openings', 'format=epd', f'file={args.book}', 'order=random',
               '-repeat',
               '-scoreinterval', str(engines * (engines - 1)),
               '-pgnout', 'nnue.pgn', 'fi']

    for nnue in args.nnue:
        command.extend(['-engine', f'name={nnue.split('/')[-1]}', f'cmd={args.bitbit}', f'option.FileNNUE={nnue}'])
    if args.reference is not None:
        command.extend(['-engine', f'name=reference', f'cmd={args.reference}'])

    print(*command, sep=' ')
    subprocess.run(command)

if __name__ == '__main__':
    main()
