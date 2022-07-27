import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import figure
from feature_generator import generate
from time import process_time
from sklearn.datasets import make_classification


def plot_class_frequency(df: pd.DataFrame, database: str):
    figure(figsize=(10, 6), dpi=300)
    labels = df[df.columns[-1]].tolist()
    x_axis = ['Have PKFK relation', 'Do not have PKFK relation']
    y_axis = [labels.count(1), labels.count(0)]
    plt.barh(x_axis, y_axis, color='darkblue')
    for index, value in enumerate(y_axis):
        plt.text(value, index,
                 str(value))
    plt.title('Class Distribution - {}'.format(database))
    plt.xlabel('Number of pairs')
    plt.tight_layout()
    plt.grid()
    plt.show()


def plot_scores(models, scores):
    figure(figsize=(10, 6), dpi=300)
    x_axis = models
    y_axis = scores
    plt.bar(x_axis, y_axis, color='darkblue', width=0.2)
    plt.title('Classifier evaluation')
    plt.xlabel('Classifiers')
    plt.ylabel('F1-score')
    plt.grid()
    plt.show()

def data_information(df: pd.DataFrame, use: str):
    labels = df[df.columns[-1]].tolist()
    print(use, ": ",labels.count(1),"/", labels.count(1)+labels.count(0))



def main():
    # Training data
    #Case 1
    #training_database_list = ['MovieLens','financial','financial_std','financial_ijs']
    #Case 2
    #training_database_list = ['MovieLens', 'tpcc']
    #Case 3
    #training_database_list = ['MovieLens', 'financial']
    #Case 4
    #training_database_list = ['MovieLens','financial','financial_std','financial_ijs', 'tpcc','SAP','SAT','Credit']
    # Case 4
    # training_database_list = ['MovieLens','financial','financial_std','TPC-H', 'tpcc','SAP','SAT','Credit']
    # Case 6
    training_database_list = ['financial']

    features = generate(training_database_list)
    # testing_database = 'MovieLens'
    # features2 = generate(testing_database)
    # features.append(features2)
    #plot_class_frequency(features, training_database)
    data_information(features, 'Training Data')
    t1_start = process_time()
    X_train = features.drop(columns=['Has_pk_fk_relation'], axis=1)
    y_train = features['Has_pk_fk_relation']

    # Testing data
    #Case 1
    #testing_database_list = ['tpcc', 'TPC-H']
    #Case 2
    #testing_database_list = ['TPC-H', 'financial', 'financial_std', 'financial_ijs']
    #Case 3
    #testing_database_list = ['TPC-H', 'financial_std', 'financial_ijs']
    #Case 4
    #testing_database_list = ['TPC-H']
    #Case 5
    #testing_database_list = ['financial_ijs']
    # Case 6
    testing_database_list = ['MovieLens']

    features = generate(testing_database_list)
    #plot_class_frequency(features, testing_database)
    data_information(features, 'Testing Data')
    X_test = features.drop(columns=['Has_pk_fk_relation'], axis=1)
    y_test = features['Has_pk_fk_relation']

    # Initiate classifiers
    svm = SVC()
    rf = RandomForestClassifier()
    nb = GaussianNB()

    # Train classifiers
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Get predictions
    y_pred_svm = svm.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_nb = nb.predict(X_test)

    # NB
    score_nb = f1_score(y_test, y_pred_nb)
    # plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall_nb = recall_score(y_test, y_pred_nb)
    precision_nb = precision_score(y_test, y_pred_nb)
    print('NB')
    print('precision: ', precision_nb)
    print('recall: ', recall_nb)
    print('F1-Score: ', score_nb)

    # SVM
    score_svm = f1_score(y_test, y_pred_svm)
    # plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall_svm = recall_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    print('SVM')
    print('precision: ', precision_svm)
    print('recall: ', recall_svm)
    print('F1-Score: ', score_svm)


    # RF
    score_rf = f1_score(y_test, y_pred_rf)
    #plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall = recall_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf)
    print('RF')
    print('precision: ',precision )
    print('recall: ', recall)
    print('F1-Score: ',score_rf)

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred_rf)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.show()
    # get importance
    importance = rf.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.4f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance[sorted_indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
    pyplot.show()


if __name__ == "__main__":
    main()

