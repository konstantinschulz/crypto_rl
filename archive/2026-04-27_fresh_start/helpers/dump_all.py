import glob

for f in sorted(glob.glob('/home/konstantin/dev/crypto_rl/manual_log*.txt')):
    print(f'=== {f} ===')
    with open(f, 'r') as file:
        lines = file.readlines()
        print(''.join(lines[:10]))
        print("...")
        print(''.join(lines[-20:]))

