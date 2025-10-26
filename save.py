import csv
from datetime import datetime

from cte import DATA_BASE


def save_results(accuracy_dict, precision_dict, recall_dict, f1_dict,strategy, add_header=True):
    header = ['QUERY', 'Accuracy', 'Precision', 'Recall', 'F1']
    with open(f'results/{DATA_BASE}/{strategy}_result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        if add_header:
            writer.writerow(header)

        writer.writerow(["----", "----", "----", "----", "----"])
        for k, v in accuracy_dict.items():
            lenn = len(v) - 1
            writer.writerow(
                [k, accuracy_dict[k][lenn], precision_dict[k][lenn], recall_dict[k][lenn], f1_dict[k][lenn]])
def save_all_iterationts_results(data, title):
    import os

    if not os.path.exists(f'results/{DATA_BASE}/'):
        os.makedirs(f'results/{DATA_BASE}/')
    with open(f'results/{DATA_BASE}/all_iterationts_results_{title}_{datetime.now().strftime("%H_%M_%S")}.csv', 'w',encoding='UTF8') as f:
        write = csv.writer(f)
        write.writerows(data)
