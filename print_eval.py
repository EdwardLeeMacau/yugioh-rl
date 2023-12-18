# Parse the metrics file and plot the results

import os

import jsonlines
import pandas as pd

OBJECTIVE = 'scores'

for experiment in os.listdir('models'):
    fpath = os.path.join('models', experiment, 'metrics.json')

    with jsonlines.open(fpath, 'r') as reader:
        metrics = [line for line in reader]
        metrics = pd.DataFrame(metrics)

    best = metrics[metrics[OBJECTIVE] == metrics[OBJECTIVE].max()]
    iloc = best.index[0]
    print(f'Experiment: {experiment}')
    print(f'Best checkpoint: {iloc}')
    print(f'Best {OBJECTIVE}: {best[OBJECTIVE].values[0]}')
    print(f'Performance at best step: {metrics.iloc[iloc].to_dict()}')
    print()