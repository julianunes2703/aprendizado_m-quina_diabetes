

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Configuração de estilo para os gráficos
sns.set_style("whitegrid")


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
colunas = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, names=colunas)

# Tratamento de zeros
cols_zero_invalidas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in cols_zero_invalidas:
    df[c] = df[c].replace(0, np.nan)
    df[c].fillna(df[c].median(), inplace=True)


#  Divisão e Balanceamento

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# SMOTE (Apenas no treino - PERFEITO!)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()

# Dados padronizados (com nomes das colunas preservados para os modelos de árvore)
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# PCA (Apenas para Regressão Logística)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Features Originais: {len(X.columns)}")
print(f"Componentes PCA: {X_train_pca.shape[1]}")

# Função auxiliar de avaliação
def avalia_modelo(nome, y_true, y_pred, y_prob):
    return [
        nome,
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_prob)
    ]

resultados = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



model_log = LogisticRegression(max_iter=1000, random_state=42)
model_log.fit(X_train_pca, y_train_res) # Treina com PCA

y_pred_log = model_log.predict(X_test_pca)
y_prob_log = model_log.predict_proba(X_test_pca)[:, 1]
resultados.append(avalia_modelo("Logistic Regression (PCA)", y_test, y_pred_log, y_prob_log))

.

# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
param_rf = {
    "n_estimators": [200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced", None] # Ajuda no recall
}
random_rf = RandomizedSearchCV(rf, param_rf, n_iter=10, scoring="recall", cv=cv, random_state=42, n_jobs=-1)
random_rf.fit(X_train_scaled, y_train_res) # Treina com SCALED (sem PCA)
best_rf = random_rf.best_estimator_

resultados.append(avalia_modelo("Random Forest", y_test, best_rf.predict(X_test_scaled), best_rf.predict_proba(X_test_scaled)[:, 1]))

# --- XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
param_xgb = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "scale_pos_weight": [1, 2] # Importante para aumentar sensibilidade (Recall)
}
random_xgb = RandomizedSearchCV(xgb, param_xgb, n_iter=10, scoring="recall", cv=cv, random_state=42, n_jobs=-1)
random_xgb.fit(X_train_scaled, y_train_res)
best_xgb = random_xgb.best_estimator_

resultados.append(avalia_modelo("XGBoost", y_test, best_xgb.predict(X_test_scaled), best_xgb.predict_proba(X_test_scaled)[:, 1]))

# --- LightGBM ---
lgbm = LGBMClassifier(random_state=42, verbosity=-1)
# LightGBM lida bem com dados raw, mas manteremos scaled por padronização
best_lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
best_lgbm.fit(X_train_scaled, y_train_res)

resultados.append(avalia_modelo("LightGBM", y_test, best_lgbm.predict(X_test_scaled), best_lgbm.predict_proba(X_test_scaled)[:, 1]))

# --- CatBoost ---
# CatBoost é excelente com dados categóricos, mas aqui só temos numéricos.
cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5, verbose=False, random_state=42)
cat.fit(X_train_scaled, y_train_res)

resultados.append(avalia_modelo("CatBoost", y_test, cat.predict(X_test_scaled), cat.predict_proba(X_test_scaled)[:, 1]))


df_resultados = pd.DataFrame(resultados, columns=["Modelo", "Acurácia", "Precisão", "Recall", "F1-score", "ROC-AUC"])
print("\nResultados Finais:")
display(df_resultados.sort_values(by="Recall", ascending=False)) # Ordenar por Recall é melhor para área médica


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
modelos_map = {
    "LogReg (PCA)": (model_log, X_test_pca),
    "Random Forest": (best_rf, X_test_scaled),
    "XGBoost": (best_xgb, X_test_scaled),
    "LightGBM": (best_lgbm, X_test_scaled),
    "CatBoost": (cat, X_test_scaled)
}

for i, (nome, (modelo, X_val)) in enumerate(modelos_map.items()):
    y_pred = modelo.predict(X_val)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
    axes[i].set_title(f"Matriz de Confusão - {nome}")
    axes[i].set_xlabel("Previsto")
    axes[i].set_ylabel("Real")
plt.tight_layout()
plt.show()


def plot_feature_importance(model, model_name, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Importância das Variáveis - {model_name}")
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
        plt.xlabel("Importância Relativa")
        plt.show()

cols = X_train_scaled.columns.tolist()

plot_feature_importance(best_rf, "Random Forest", cols)
plot_feature_importance(best_xgb, "XGBoost", cols)
plot_feature_importance(best_lgbm, "LightGBM", cols)
plot_feature_importance(cat, "CatBoost", cols)
