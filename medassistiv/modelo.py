# bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# lendo o dataset e removendo duplicatas
df = pd.read_csv("velhos.csv").drop_duplicates()

# binariza a coluna de risco (1 = High Risk, 0 = Low Risk)
df['Normalized_risk'] = (df['Risk Category'] == 'High Risk').astype(int)

# features usadas
X = df[['Heart Rate', 'Body Temperature', 'Oxygen Saturation',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure',
        'Age', 'Weight (kg)', 'Height (m)', 'Derived_BMI', 'Respiratory Rate']]

y = df['Normalized_risk']

# separa treino e teste (estratificado para manter proporção de classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# lida com possível desbalanceamento
neg, pos = np.bincount(y_train)
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

# modelo base (SEM use_label_encoder)
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',      # rápido e estável
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# grade de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Melhor pontuação de validação:", grid.best_score_)
print("Melhores parâmetros:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=4))

joblib.dump(best_model, "modelo_xgboost.joblib")
print("\nModelo salvo como 'modelo_xgboost.joblib'")
