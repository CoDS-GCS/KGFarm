import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import figure
from feature_generator import generate
from time import process_time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
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
    #train test split
    database_list = ['TPC-H', 'MovieLens', 'financial', 'financial_std', 'financial_ijs']
    features = generate(database_list)
    data_information(features, 'Testing Data')
    X = features.drop(columns=['Has_pk_fk_relation'], axis=1)
    y = features['Has_pk_fk_relation']

    X_under, X_test, y_under, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12,stratify=y)
    print(Counter(y_under))
    # Undersampling
    undersample = RandomUnderSampler(sampling_strategy=0.33)
    X_train, y_train = undersample.fit_resample(X_under, y_under)
    print(Counter(y_under))

    print(y_train.value_counts())
    print('-----------------------')
    print(y_test.value_counts())

    #Grid Search for SVM
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # Initiate classifiers1
    svm = SVC()
    rf = RandomForestClassifier(class_weight="balanced")
    nb = GaussianNB()

    # Train classifiers
    nb.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Get predictions
    y_pred_svm = grid.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred_nb = nb.predict(X_test)

    # NB
    score_nb = f1_score(y_test, y_pred_nb)
    # plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall_nb = recall_score(y_test, y_pred_nb)
    precision_nb = precision_score(y_test, y_pred_nb)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print('----------------------------------')
    print('NB')
    print('accuracy: %.3f'% accuracy_nb)
    print('precision: %.3f'% precision_nb)
    print('recall: %.3f'% recall_nb)
    print('F1-Score: %.3f' % score_nb)

    # SVM
    score_svm = f1_score(y_test, y_pred_svm)
    # plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall_svm = recall_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print('----------------------------------')
    print('SVM')
    print('accuracy:%.3f ' % accuracy_svm)
    print('precision: %.3f '% precision_svm)
    print('recall: %.3f ' %recall_svm)
    print('F1-Score: %.3f '% score_svm)


    # RF
    score_rf = f1_score(y_test, y_pred_rf)
    #plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall = recall_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print('----------------------------------')
    print('RF')
    print('accuracy: %.3f'% accuracy_rf)
    print('precision: %.3f'%precision )
    print('recall: %.3f'% recall)
    print('F1-Score: %.3f'%score_rf)

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

