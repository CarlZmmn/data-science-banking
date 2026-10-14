import pandas as pd
from IPython.display import display
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix)
import time
import pickle

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
    abgelehnt = tp + fp
    ablehnung_unnoetig = fp / abgelehnt if abgelehnt > 0 else float('nan')   # Anteil der Abgelehnten, die gezahlt hätten
    gute_verloren = fp / (tn + fp) if (tn + fp) > 0 else float('nan')        # Anteil aller guten Kunden, die abgelehnt wurden
    ziel = 'erreicht' if ausfall_genehmigt < ZIEL_AUSFALLQUOTE else 'nicht erreicht'

    print(f'{name}')
    print(f'Genehmigungsquote {genehmigungsquote:.1%} | Ausfallquote genehmigt {ausfall_genehmigt:.1%} '
         #f'→ Ziel < {ZIEL_AUSFALLQUOTE:.0%} {ziel}'
          )
    print(f'Abgelehnt, hätten aber gezahlt: {fp} von {abgelehnt} Ablehnungen ({ablehnung_unnoetig:.1%}) '
          f'| {gute_verloren:.1%} aller guten Kunden abgelehnt')
    print(
        f'erkannte Ausfälle {recall:.1%}')
    print(f'\nPrecision {precision:.3f}\nRecall {recall:.3f}'
          f'\nF1 {f1:.3f}\nMCC {mcc:.3f}' + (f'\nROC AUC {roc_auc:.3f}' if roc_auc is not None else ''))
    display(pd.DataFrame(cm,
                         index=['true: zahlt', 'true: Ausfall'],
                         columns=['pred.: zahlt', 'pred.: Ausfall']))
    print()


def trainieren_und_bewerten(name, model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    zeit_fit = time.perf_counter() - start                          # Trainingszeit in Sekunden

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    zeit_predict = time.perf_counter() - start                      # Vorhersagezeit für das ganze Testset

    if hasattr(model, 'predict_proba'):                             # Wahrscheinlichkeiten für ROC AUC
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):                       # z. B. SVC ohne probability=True
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    groesse_mb = len(pickle.dumps(model)) / 1e6                     # Speicherbedarf des trainierten Modells

    bewerten(name, y_test, y_pred, y_score)
    print(f'Ressourcen | Training {zeit_fit:.2f} s | Vorhersage {zeit_predict * 1000:.0f} ms '
          f'({zeit_predict / len(X_test) * 1e6:.1f} µs pro Person) | Modellgröße {groesse_mb:.2f} MB\n')
    return model