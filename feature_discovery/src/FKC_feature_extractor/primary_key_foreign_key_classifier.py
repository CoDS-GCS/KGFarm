import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.pyplot import figure
from feature_generator import generate


def plot_class_frequency(df: pd.DataFrame):
    figure(figsize=(10, 6), dpi=300)
    labels = df[df.columns[-1]].tolist()
    x_axis = ['Have PKFK relation', 'Do not have PKFK relation']
    y_axis = [labels.count(1), labels.count(0)]
    plt.barh(x_axis, y_axis, color='darkblue')
    for index, value in enumerate(y_axis):
        plt.text(value, index,
                 str(value))
    plt.title('Class Distribution')
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
    features = generate()
    plot_class_frequency(features)
    X = features.drop(columns=['Has_pk_fk_relation', 'F3'], axis=1)
    y = features['Has_pk_fk_relation']
    svm = SVC()
    rf = RandomForestClassifier()
    scores_svm = np.mean(cross_val_score(svm, X, y, cv=5, scoring='f1'))
    scores_rf = np.mean(cross_val_score(rf, X, y, cv=5, scoring='f1'))
    plot_scores(['SVM Classifier', 'Random Forest Classifier'], [scores_svm, scores_rf])


if __name__ == "__main__":
    main()

