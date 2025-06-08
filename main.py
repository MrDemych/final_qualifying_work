# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Загрузка данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
data = pd.read_csv(url)

# Предобработка
data = data.drop(columns=['UDI', 'Product ID'])
data['Type'] = LabelEncoder().fit_transform(data['Type'])
scaler = StandardScaler()
numeric_cols = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Разделение данных
X = data.drop('Machine failure', axis=1)
y = data['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение моделей
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if name != "XGBoost" else model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "model": model
    }
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

# Вывод результатов
print("Результаты оценки моделей:")
for model, metrics in results.items():
    print(f"\n{model}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    plt.figure()
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {model}')
    plt.savefig(f"{model.replace(' ', '_')}_conf_matrix.png")
    plt.close()