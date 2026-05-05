import glob
import os

for f in sorted(glob.glob('manual_log*.txt')):
    print(f'=== {f} ===')
    with open(f, 'r') as file:
        lines = file.readlines()
        if len(lines) > 0:
            print(''.join(lines[-35:]))
        else:
            print("EMPTY")

