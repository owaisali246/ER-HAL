from copy import deepcopy
import pandas as pd
import os

from Result import Result
from save import save_results
from cte import NI, DPQ, STQ
from data_reading import get_data
from ActiveLearning import start_active_learning
from tools import add_result


def get_init_dicts(strategy):
    accuracy_dict, precision_dict, recall_dict, f1_dict = {}, {}, {}, {}
    accuracy_dict[strategy] = []
    precision_dict[strategy] = []
    recall_dict[strategy] = []
    f1_dict[strategy] = []
    return accuracy_dict, precision_dict, recall_dict, f1_dict


def my_function(strategy, x_pool, y_pool, x_test, y_test):
    learner, scores = start_active_learning(strategy, x_pool, y_pool, x_test, y_test)
    return Result(strategy, learner, scores)


def get_query_strategy():
    choice = int(input("choose a strategy (1) for DPQ, (2) for STQ  :    \t"))
    if choice == 1:
        return DPQ
    elif choice == 2:
        return STQ
    raise ValueError(f'Error:the query strategy [{choice}] is not supported ')


def main():
    x_train, y_train, x_test, y_test = get_data()
    strategy = get_query_strategy()
    accuracy_dict, precision_dict, recall_dict, f1_dict = get_init_dicts(strategy)

    all_metrics = []

    for i in range(NI):
        print("ni =", i)
        x_pool = x_train
        y_pool = y_train

        result = my_function(
            strategy,
            deepcopy(x_pool),
            deepcopy(y_pool),
            deepcopy(x_test),
            deepcopy(y_test),
        )

        accuracy_dict, precision_dict, recall_dict, f1_dict = add_result(
            accuracy_dict, precision_dict, recall_dict, f1_dict, [result]
        )

        acc_values = accuracy_dict[strategy][0]
        prec_values = precision_dict[strategy][0]
        rec_values = recall_dict[strategy][0]
        f1_values = f1_dict[strategy][0]

        for j in range(len(f1_values)):
            all_metrics.append({
                "iteration": i,
                "step": j,
                "accuracy": acc_values[j],
                "precision": prec_values[j],
                "recall": rec_values[j],
                "f1_score": f1_values[j]
            })

    df = pd.DataFrame(all_metrics)
    os.makedirs("result_graph", exist_ok=True)
    csv_path = f"result_graph/results_{strategy}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to {csv_path}")

    print(df.head())

    save_results(accuracy_dict, precision_dict, recall_dict, f1_dict, strategy)


if __name__ == "__main__":
    main()
