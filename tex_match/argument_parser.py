import argparse
import os
from .constants import PARSE_DIR, OUTPUT_DIR, SUPPORTED_FORMATS

def parse_args():
    parser = argparse.ArgumentParser(
        description='Command line tool for matching texture maps to the provided input map. ' \
                    'Returns a plain text file with possible matches.'
    )
    parser.add_argument(
        'input',
        type=str,
        help='path to the input texture map'
    )
    parser.add_argument(
        '-p',
        '--parse-dir',
        default=PARSE_DIR,
        help='directory to look for matches in'
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        default=OUTPUT_DIR,
        help='directory to write matches to'
    )
    parser.add_argument(
        '-t',
        action='store_true',
        help='use template matching (fast, default)'
    )
    parser.add_argument(
        '-f',
        action='store_true',
        help='match based on local features (slow)'
    )

    args = parser.parse_args()
    verify_args(args)

    return args

def verify_args(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError('The input file does not exist.')
    else:
        name, extension = os.path.splitext(args.input)
        if extension not in SUPPORTED_FORMATS:
            raise UserWarning('The input file is of an unsupported format.')

    if not os.path.isdir(args.parse_dir):
        raise NotADirectoryError('The directory to parse does not exist.')
    elif not os.listdir(args.parse_dir):
        raise UserWarning('The directory to parse is empty.')

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)