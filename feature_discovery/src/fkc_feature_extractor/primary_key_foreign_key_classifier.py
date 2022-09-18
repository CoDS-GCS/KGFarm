import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from feature_generator import generate


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


def main():
    # Training data
    training_database = 'TPC-H'
    features = generate(training_database)
    plot_class_frequency(features, training_database)
    X_train = features.drop(columns=['Has_pk_fk_relation', 'F3'], axis=1)
    y_train = features['Has_pk_fk_relation']

    # Testing data
    testing_database = 'MovieLens'
    features = generate(testing_database)
    plot_class_frequency(features, testing_database)
    X_test = features.drop(columns=['Has_pk_fk_relation', 'F3'], axis=1)
    y_test = features['Has_pk_fk_relation']

    # Initiate classifiers
    svm = SVC()
    rf = RandomForestClassifier()

    # Train classifiers
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Get predictions
    y_pred_svm = svm.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Plot f1
    score_svm = f1_score(y_test, y_pred_svm)
    score_rf = f1_score(y_test, y_pred_rf)
    plot_scores(['SVM Classifier', 'Random Forest Classifier'], [score_svm, score_rf])


if __name__ == "__main__":
    main()

