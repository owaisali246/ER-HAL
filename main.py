from copy import deepcopy

from Result import Result
from save import save_results
from cte import NI, DPQ, STQ
from data_reading import get_data
from ActiveLearning import start_active_learning
from tools import add_result, get_average_graph


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
    # strategy = get_query_strategy()
    strategy = STQ
    accuracy_dict, precision_dict, recall_dict, f1_dict = get_init_dicts(strategy)
    for i in range(NI):
        print("ni=", i)
        x_pool = x_train
        y_pool = y_train
        results = []

        results.append(my_function(strategy, deepcopy(x_pool), deepcopy(y_pool), deepcopy(x_test), deepcopy(y_test)))

        # save_results(results)
        accuracy_dict, precision_dict, recall_dict, f1_dict = add_result(accuracy_dict, precision_dict,
                                                                         recall_dict, f1_dict, results)
    get_average_graph(f1_dict, "F1")
    get_average_graph(accuracy_dict, "Accuracy")
    get_average_graph(precision_dict, "Precision")
    get_average_graph(recall_dict, "Recall")
    save_results(accuracy_dict, precision_dict, recall_dict, f1_dict, strategy)

if __name__ == "__main__":
    main()
