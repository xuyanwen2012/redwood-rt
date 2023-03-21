import subprocess
import argparse
import tempfile
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cpu', action='store_true',
                    help='enable cpu')
args = parser.parse_args()

binary = './cuda.out'
datafile = '../../data/input_nn_1m_4f.dat 1048576'
l_values = [32, 64, 128, 256, 512, 1024]

for l in l_values:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        command = f'{binary} {datafile} -l {l}'
        if args.cpu:
            command += " -c"

        subprocess.run(command, shell=True, stdout=temp_file)

        temp_file.seek(0)
        output_lines = temp_file.readlines()

        time_regex = re.compile(
            r'Finished Traversal! Time took: (\d+\.\d+)s.')
        for line in output_lines:
            match = time_regex.search(line)
            if match:
                time_taken = float(match.group(1))
                print(f'Time took for -l {l}: {time_taken}s')

        os.remove(temp_file.name)
