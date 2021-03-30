import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./datasets/heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv')
    df.columns
    features = df.drop(['target'], axis=1)

    target = df['target']

    x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=.35)

    kn = KNeighborsClassifier().fit(x_train,y_train)
    kn_pred = kn.predict(x_test)
    print(accuracy_score(kn_pred,y_test))

    arboles = DecisionTreeClassifier().fit(x_train,y_train)
    arboles_pred = arboles.predict(x_test)
    print(accuracy_score(arboles_pred,y_test))

    bagg = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=400).fit(x_train,y_train)
    bagg_pred = bagg.predict(x_test)
    print(accuracy_score(bagg_pred,y_test))

    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.35)

    estimators = range(10, 300, 10)
    total_accuracy = []
    for i in estimators:
        boost = GradientBoostingClassifier(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)

        total_accuracy.append(accuracy_score(y_test, boost_pred))

    print(max(total_accuracy))
    plt.plot(estimators,total_accuracy)
    plt.show()


    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    GradientBoostingClassifier(criterion='friedman_mse', init=None,
    learning_rate=0.1, loss='deviance', max_depth=3,
    max_features=None, max_leaf_nodes=None,
    min_impurity_split=1e-07, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    n_estimators=100, presort='auto', random_state=None,
    subsample=1.0, verbose=0, warm_start=False)
    y_pred = model.predict(x_test)

    from sklearn.metrics import roc_curve, auc

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print(roc_auc)

    learning_rates = range(1,200,10)
    train_results = []
    test_results = []
    for eta in learning_rates:
        model = GradientBoostingClassifier(learning_rate=eta)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = model.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(learning_rates, train_results, 'b', label='Train AUC')
    line2, = plt.plot(learning_rates, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('learning rate')
    plt.show()

