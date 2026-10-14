import pandas as pd
from IPython.display import display
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix)

ZIEL_AUSFALLQUOTE = 0.085   # max. Anteil Ausfälle unter den genehmigten Krediten


def bewerten(name, y_true, y_pred, y_score=None):
    accuracy = accuracy_score(y_true, y_pred)                        # Anteil richtiger Vorhersagen
    precision = precision_score(y_true, y_pred, zero_division=0)     # Von den abgelehnten: wie viele echte Ausfälle?
    recall = recall_score(y_true, y_pred)                            # Von den echten Ausfällen: wie viele erkannt?
    f1 = f1_score(y_true, y_pred)                                    # Kombination aus Precision und Recall
    mcc = matthews_corrcoef(y_true, y_pred)                          # Gesamtqualität, fair bei ungleichen Klassen
    roc_auc = roc_auc_score(y_true, y_score) if y_score is not None else None  # Trennschärfe über alle Schwellen

    # Ausfallquote unter den genehmigten Krediten = FN / (TN + FN)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    genehmigt = tn + fn
    genehmigungsquote = genehmigt / len(y_true)
    ausfall_genehmigt = fn / genehmigt if genehmigt > 0 else float('nan')
    ziel = 'erreicht' if ausfall_genehmigt < ZIEL_AUSFALLQUOTE else 'nicht erreicht'

    print(f'{name}')
    print(f'Genehmigungsquote {genehmigungsquote:.1%} | Ausfallquote genehmigt {ausfall_genehmigt:.1%} '
         #f'→ Ziel < {ZIEL_AUSFALLQUOTE:.0%} {ziel}'
          )
    print(f'Fehlerquote {1 - accuracy:.1%} | erkannte Ausfälle {recall:.1%}')
    print(f'\nAccuracy {accuracy:.3f}\nPrecision {precision:.3f}\nRecall {recall:.3f}'
          f'\nF1 {f1:.3f}\nMCC {mcc:.3f}' + (f'\nROC AUC {roc_auc:.3f}' if roc_auc is not None else ''))
    display(pd.DataFrame(cm,
                         index=['true: zahlt', 'true: Ausfall'],
                         columns=['pred.: zahlt', 'pred.: Ausfall']))
    print()
