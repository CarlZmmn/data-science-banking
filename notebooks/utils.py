import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from IPython.display import display

def bewerten(name, y_true, y_pred):
    fehler = 1 - accuracy_score(y_true, y_pred)               # Fehlerquote der tatsächlichen Vorhersagen
    erkannt = recall_score(y_true, y_pred)                  # Anteil der tatsächlich erkannten Ausfälle
    print(f'{name}| Fehlerquote {fehler:.1%} | erkannte Ausfälle {erkannt:.1%}')
    display(pd.DataFrame(confusion_matrix(y_true, y_pred),
                       index=['true: zahlt', 'true: Ausfall'],
                       columns=['pred.: zahlt', 'pred.: Ausfall']))
    print()