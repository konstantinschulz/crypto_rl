import json, glob
f = sorted(glob.glob('logs/run-*/actions_eval_*.jsonl'))[-1]
lines = [json.loads(l) for l in open(f)]
from collections import Counter
print(Counter(l['action_type'] for l in lines))
print('Final portfolio:', lines[-1]['portfolio'])
