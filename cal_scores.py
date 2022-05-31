import os
import numpy as np

data_name = 'snips'
assert data_name in ['snips', 'stsa', 'trec']

DA_method = 't5'
assert DA_method in ['baseline', 'bt', 'cbert',
                     'eda', 'cmodbert', 'cmodbertp', 't5']

base_dir = f"datasets/{data_name}"

dev_acc = []
test_acc = []
for dir_name in os.listdir(base_dir):
    if dir_name.endswith("tsv"):
        continue

    with open(f"{base_dir}/{dir_name}/bert_{DA_method}.log", "r") as f:
        for line in f:
            pass
        last_line = line.split()
        dev_acc.append(float(last_line[3][:-1]))
        test_acc.append(float(last_line[-1]))

print(np.mean(dev_acc), np.std(dev_acc))
print(np.mean(test_acc), np.std(test_acc))
