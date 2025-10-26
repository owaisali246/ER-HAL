from modAL import ActiveLearner, Committee
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier

from Extending_modAL import custom_query_strategy
random_forest_estimator = RandomForestClassifier()
ESTIMATOR = random_forest_estimator


def get_active_learner(strategy):
    ENEMBLE_LEARNING_no_BOOTSTRAP = [
        # DecisionTreeClassifier(max_depth=10),
        # RandomForestClassifier(),
        RandomForestClassifier(criterion='entropy'),
        # MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()
    ]
    learner_list = []


    for clf in ENEMBLE_LEARNING_no_BOOTSTRAP:
        learner_list.append(ActiveLearner(estimator=clf))
    Committee(learner_list)
    return get_active_learnerCTE(ESTIMATOR)

def get_active_learnerCTE(es):
    return ActiveLearner(estimator=es, query_strategy=custom_query_strategy)

