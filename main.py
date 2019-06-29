import pandas as pd
from sklearn import metrics

classification = pd.read_csv('classification.csv')
TP = FP = FN = TN = 0
for index, row in classification.iterrows():
    if row['true'] == 1 and row['pred'] == 1:
        TP += 1
    elif row['true'] == 0 and row['pred'] == 1:
        FP += 1
    elif row['true'] == 0 and row['pred'] == 0:
        TN += 1
    else:
        FN += 1

accuracy = metrics.accuracy_score(classification['true'], classification['pred'])
precision = metrics.precision_score(classification['true'], classification['pred'])
recall = metrics.recall_score(classification['true'], classification['pred'])
F = metrics.f1_score(classification['true'], classification['pred'])

scores = pd.read_csv('scores.csv')
logreg = metrics.roc_auc_score(scores['true'], scores['score_logreg'])
svm = metrics.roc_auc_score(scores['true'], scores['score_svm'])
knn = metrics.roc_auc_score(scores['true'], scores['score_knn'])
tree = metrics.roc_auc_score(scores['true'], scores['score_tree'])

max_logreg_precision = 0.0
prec, rec, thresh = metrics.precision_recall_curve(scores['true'], scores['score_logreg'])
for i in range(len(prec)):
    if rec[i] > 0.7 and prec[i] > max_logreg_precision:
        max_logreg_precision = prec[i]

max_svm_precision = 0.0
prec, rec, thresh = metrics.precision_recall_curve(scores['true'], scores['score_svm'])
for i in range(len(prec)):
    if rec[i] > 0.7 and prec[i] > max_svm_precision:
        max_svm_precision = prec[i]

max_knn_precision = 0.0
prec, rec, thresh = metrics.precision_recall_curve(scores['true'], scores['score_knn'])
for i in range(len(prec)):
    if rec[i] > 0.7 and prec[i] > max_knn_precision:
        max_knn_precision = prec[i]

max_tree_precision = 0.0
prec, rec, thresh = metrics.precision_recall_curve(scores['true'], scores['score_tree'])
for i in range(len(prec)):
    if rec[i] > 0.7 and prec[i] > max_tree_precision:
        max_tree_precision = prec[i]
