import joblib
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, \
    accuracy_score, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import figure
from feature_generator import generate
from time import process_time
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, KFold
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

def get_importance(rf, X_train):
    importance = rf.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.4f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance[sorted_indices], align='center')
    plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
    pyplot.show()

def get_results_without_kfold(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(X_train)
    print(X_test)
    print(Counter(y_train))

    print(y_train.value_counts())
    print('-----------------------')
    print(y_test.value_counts())

    # Grid Search for SVM
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']}
    #
    # grid = GridSearchCV(SVC(class_weight="balanced",random_state=42), param_grid, refit=True)
    #
    # # fitting the model for grid search
    # grid.fit(X_train, y_train)

    # Initiate classifiers1
    svm = SVC(random_state=42)
    rf = RandomForestClassifier(class_weight="balanced",random_state=42)
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
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print('----------------------------------')
    print('NB')
    print('accuracy: %.3f' % accuracy_nb)
    print('precision: %.3f' % precision_nb)
    print('recall: %.3f' % recall_nb)
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
    print('precision: %.3f ' % precision_svm)
    print('recall: %.3f ' % recall_svm)
    print('F1-Score: %.3f ' % score_svm)

    # RF
    score_rf = f1_score(y_test, y_pred_rf)
    # plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])
    recall = recall_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print('----------------------------------')
    print('RF')
    print('accuracy: %.3f' % accuracy_rf)
    print('precision: %.3f' % precision)
    print('recall: %.3f' % recall)
    print('F1-Score: %.3f' % score_rf)


def get_results_with_kfold(X, y):
    # Set up k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize classifiers
    nb_classifier = GaussianNB()
    svm_classifier = SVC(random_state=42)
    rf_classifier = RandomForestClassifier(random_state=42)

    # Run k-fold cross-validation and get predictions
    nb_predictions = cross_val_predict(nb_classifier, X, y, cv=kfold)
    svm_predictions = cross_val_predict(svm_classifier, X, y, cv=kfold)
    rf_predictions = cross_val_predict(rf_classifier, X, y, cv=kfold)

    # Calculate accuracy and F1 score for Naive Bayes
    nb_accuracy = accuracy_score(y, nb_predictions)
    nb_f1 = f1_score(y, nb_predictions, average='weighted')

    # Calculate accuracy and F1 score for SVM
    svm_accuracy = accuracy_score(y, svm_predictions)
    svm_f1 = f1_score(y, svm_predictions, average='weighted')

    # Calculate accuracy and F1 score for Random Forest
    rf_accuracy = accuracy_score(y, rf_predictions)
    rf_f1 = f1_score(y, rf_predictions, average='weighted')

    # Display the results
    print("Naive Bayes Accuracy:", nb_accuracy)
    print("Naive Bayes F1 Score:", nb_f1)

    print("\nSVM Accuracy:", svm_accuracy)
    print("SVM F1 Score:", svm_f1)

    print("\nRandom Forest Accuracy:", rf_accuracy)
    print("Random Forest F1 Score:", rf_f1)

    # #Grid Search for SVM
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']}
    #
    # # Initiate classifiers1
    # svm = SVC()
    # rf = RandomForestClassifier(class_weight="balanced")
    # nb = GaussianNB()
    #
    # classifiers = [svm, rf, nb]
    #
    # for classifier in classifiers:
    #     print('Using ',classifier)
    #     inner_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    #     outer_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
    #     if classifier == svm:
    #         classifier = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=inner_cv)
    #     #print(cross_val_score(estimator=classifier, X=X, y=y, cv=outer_cv, scoring='f1',error_score='raise'))
    #     pred = cross_val_predict(classifier, X, y, cv=outer_cv)
    #     print(classification_report(y_true=y, y_pred=pred,labels=[0,1]))

def get_results_using_model(X,y):
    # Load the model from the pickle file
    rf = joblib.load('pk-fk_sub_w_f3.pkl')
    # Check if the model is an instance of RandomForestClassifier
    if isinstance(rf, RandomForestClassifier):
        # Get feature importances
        feature_importances = rf.feature_importances_

        # Create a DataFrame to display feature names and their importance
        importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

        # Sort the DataFrame by importance in descending order
        importances_df = importances_df.sort_values(by='Importance', ascending=False)

        # Display the feature importances
        print(importances_df)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_predictions = cross_val_predict(rf, X, y, cv=kfold)

    # Calculate accuracy and F1 score for Random Forest
    rf_accuracy = accuracy_score(y, rf_predictions)
    rf_f1 = f1_score(y, rf_predictions, average='weighted')
    overall_report = classification_report(y, rf_predictions)
    print(f"Overall Classification Report:\n{overall_report}")
    overall_matrix = confusion_matrix(y, rf_predictions)
    print(f"Overall Confusion Matrix:\n{overall_matrix}")
    print("\nRandom Forest Accuracy:", rf_accuracy)
    print("Random Forest F1 Score:", rf_f1)


def Save_results(X, y):
    rf = RandomForestClassifier(class_weight="balanced")
    print(X,y)
    rf.fit(X, y)
    joblib.dump(rf, 'pk-fk_sub_w_f3.pkl', compress=9)#'SAP','tpcd','tpcds','Credit','Grants'

def main():
    #train test split
    # database_list = ['SAP','tpcd','tpcds','Credit','Grants']#['TPC-H', 'financial', 'financial_std', 'financial_ijs','cs', 'SAT','tpcc']#'joinability_study']#['TPC-H_old', 'financial_old', 'TPC-H', 'financial', 'financial_std', 'financial_ijs','cs', 'SAT']#'MovieLens-09','SAT-15']#'MovieLens','SAT','SAP','TPC-H','Basketball_men', 'financial', 'financial_std', 'financial_ijs','cs']#,'tpcc'
    database_list = [ 'financial', 'financial_std', 'financial_ijs', 'cs', 'tpcc','tpcd', 'tpch','Credit','Grants']#'SAP','SAT'

    for i, item in enumerate(database_list):
        # Print all items except the current one
        remaining_items = [x for j, x in enumerate(database_list) if j != i]
        features_train = generate(remaining_items)
        features_test = generate([item])
        X = features_train.drop(columns=['Has_pk_fk_relation'], axis=1)
        y = features_train['Has_pk_fk_relation']
        Save_results(X, y)
        X = features_test.drop(columns=['Has_pk_fk_relation'], axis=1)
        y = features_test['Has_pk_fk_relation']
        get_results_using_model(X, y)

    #
    # total_pkfk_samples = features.loc[features['Has_pk_fk_relation'] == 1]
    # print("Total number of samples: ",len(total_pkfk_samples))
    #
    # total_not_pkfk_samples = features.loc[features['Has_pk_fk_relation'] == 0]
    # print("Total number of non pk-fk samples: ", len(total_not_pkfk_samples))
    #
    # #undersampling
    # not_pkfk_samples = total_not_pkfk_samples.sample(n=len(total_pkfk_samples) * 3)
    # print("not pf-fk samples after undersampling: ", len(not_pkfk_samples))
    #
    # #merge pk-fk pairs and non pk-fk pairs
    # data = pd.concat([total_pkfk_samples, not_pkfk_samples])
    # data = data.sample(frac=1)
    # data_information(features, 'Testing Data')
    # features = generate(database_list)
    # X = features.drop(columns=['Has_pk_fk_relation'], axis=1)
    # y = features['Has_pk_fk_relation']
    # Save_results(X, y)
    # X = features_test.drop(columns=['Has_pk_fk_relation'], axis=1)
    # y = features_test['Has_pk_fk_relation']
    # get_results_using_model(X,y)
    # get_results_without_kfold(X,y)
    # get_results_with_kfold(X, y)
    #
    # Save_results(X, y)

    # Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred_rf)
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.show()

    # get importance
    #get_importance(rf, X_train)


if __name__ == "__main__":
    main()

