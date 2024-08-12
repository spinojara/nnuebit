#!/usr/bin/env python3

import argparse
import subprocess
import sys

from tcadjust.tcadjust import tcadjust

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bitbit', type=str, help='bitbit binary')
    parser.add_argument('book', type=str, help='Opening book (.epd)')
    parser.add_argument('concurrency', type=int, help='Concurrency')
    parser.add_argument('--tc', type=str, help='Time control', default='40/5+0.05')
    parser.add_argument('nnue', type=str, nargs='+', help='NNUE files')

    args = parser.parse_args()

    engines = len(args.nnue)

    if engines < 2:
        print('Input at least two NNUE networks.')
        sys.exit(1)

    command = ['cutechess-cli',
               '-tournament', 'round-robin',
               '-games', '2',
               '-rounds', '10000',
               '-concurrency', str(args.concurrency),
               '-each', 'proto=uci', f'tc={tcadjust(args.tc)}',
               '-openings', 'format=epd', f'file={args.book}', 'order=random',
               '-repeat',
               '-pgnout', 'nnue.pgn', 'fi']

    for nnue in args.nnue:
        command.extend(['-engine', f'name={nnue}', f'cmd={args.bitbit}', f'option.FileNNUE={nnue}'])

    print(*command, sep=' ')
    subprocess.run(command)

if __name__ == '__main__':
    main()
