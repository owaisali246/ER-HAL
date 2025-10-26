import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

from Score import Score


def get_diff_class(data):
    if np.unique(data).size == 2:
        return False
    return True


def get_mesures(learner, x_test, y_test):
    y_predicted = learner.predict(x_test)

    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, zero_division=0)
    recall = recall_score(y_test, y_predicted, zero_division=0)
    f1 = f1_score(y_test, y_predicted, zero_division=0)
    score = Score(accuracy, precision, recall, f1)
    return score


def add_result(accuracy_dict, precision_dict, recall_dict, f1_dict, results):
    for result in results:
        accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
        for score in result.scores_data:
            accuracy_scores.append(score.accuracy)
            precision_scores.append(score.precision)
            recall_scores.append(score.recall)
            f1_scores.append(score.f1)
        accuracy_dict[result.strategy].append(accuracy_scores)
        precision_dict[result.strategy].append(precision_scores)
        recall_dict[result.strategy].append(recall_scores)
        f1_dict[result.strategy].append(f1_scores)
    return accuracy_dict, precision_dict, recall_dict, f1_dict


def get_average_graph(dict, title):
    for k, v in dict.items():
        column_avg = np.mean(np.array(v), axis=0)
        dict[k] = list(np.round(column_avg, 3))
