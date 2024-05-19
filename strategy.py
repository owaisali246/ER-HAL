import math

from cte import N_QUERIES, DPQ, STQ


def DPQ_value(i):
    if i < 5 * N_QUERIES / 100:
        return 1
    return 0


def STQ_value(i):
    c = 5000
    return math.exp(-c * i / N_QUERIES)


def get_omega_value(strategy, i):
    if strategy == DPQ:
        return DPQ_value(i)
    elif strategy == STQ:
        return STQ_value(i)
    else:
        raise ValueError("Error : Query strategy not supported")
