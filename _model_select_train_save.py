import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn import metrics
import matplotlib.pyplot as plt


def print_anythig(y_true, y_pred):
    print(y_true.tolist())
    print(y_pred.tolist())
    print(':', np.mean(y_true == y_pred))
    print('recall :', metrics.recall_score(y_true, y_pred, average='macro'))
    print('precision :', metrics.precision_score(y_true, y_pred, average='macro'))
    print('f1_score :', metrics.f1_score(y_true, y_pred, average='macro'))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=3)
    print(fpr)
    print(tpr)
    print('AUC :', metrics.auc(fpr, tpr))
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.plot(fpr, tpr)
    plt.title('ROC曲线')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.show()


def scores(train, test):
    models = [LogisticRegression(multi_class='multinomial', solver='saga'), MultinomialNB(), KNeighborsClassifier(),
              DecisionTreeClassifier(criterion='entropy'), DecisionTreeClassifier(), RandomForestClassifier(),
              XGBClassifier(n_estimators=20, learning_rate=0.1,
                            max_depth=5,
                            min_child_weight=1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='multi:softmax',
                            num_class=3,
                            nthread=4,
                            seed=27),
              AdaBoostClassifier(n_estimators=100)]
    model_names = ['Logistic', 'MultinomialNB', 'KNN', 'id3', 'cater', 'RandomForest', 'xgboost', 'adaboost']
    for i in range(len(models)):
        model = models[i]
        score = cross_val_score(model, train, test, cv=10)
        label = model_names[i]
        print(label, ' : ', score.mean(), score)
        plt.plot(range(1, 11), score, label=label)
        plt.legend()
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('./preprocess_data/train_f_g.csv')
    print(data.head())
    col_tr = data.columns.tolist()
    col_tr.remove('label')
    train = data[col_tr]
    test = data['label']

    # 交叉验证
    scores(train, test)

    # model = AdaBoostClassifier(n_estimators=100)
    # model.fit(train, test)
    #
    # # 模型保存
    # if not os.path.exists('./model'):
    #     os.makedirs('./model')
    # with open('./model/model.pickle', 'wb') as f:
    #     pickle.dump(model, f)
