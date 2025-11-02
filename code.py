

# 1. Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 2. Carregar dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
colunas = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=colunas)

print("Primeiras linhas:")
display(df.head())

# 3. Verificar dados ausentes ou zeros inválidos
cols_zero_invalidas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for c in cols_zero_invalidas:
    df[c] = df[c].replace(0, np.nan)
    df[c].fillna(df[c].median(), inplace=True)

print("\nResumo estatístico após limpeza:")
display(df.describe())

# 4. Separar features e target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Divisão treino/teste (Holdout)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 6. Modelos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# 7. Treinamento e avaliação
resultados = []

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    resultados.append([nome, acc, prec, rec, f1, auc])

df_resultados = pd.DataFrame(resultados, columns=["Modelo", "Acurácia", "Precisão", "Recall", "F1-score", "ROC-AUC"])
print("\nResultados Iniciais:")
display(df_resultados.sort_values(by="ROC-AUC", ascending=False))

#gráfico da importancia das variaveis

# 1. Random Forest
rf_importance = modelos["Random Forest"].feature_importances_
rf_feature_names = X.columns
rf_df = pd.DataFrame({
    'Feature': rf_feature_names,
    'Importance': rf_importance
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=rf_df)
plt.title("Importância das Variáveis - Random Forest")
plt.show()

# 2. XGBoost
xgb_importance = modelos["XGBoost"].feature_importances_
xgb_df = pd.DataFrame({
    'Feature': rf_feature_names,
    'Importance': xgb_importance
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=xgb_df)
plt.title("Importância das Variáveis - XGBoost")
plt.show()

# 3. LightGBM
lgbm_importance = modelos["LightGBM"].booster_.feature_importance()
lgbm_df = pd.DataFrame({
    'Feature': rf_feature_names,
    'Importance': lgbm_importance
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=lgbm_df)
plt.title("Importância das Variáveis - LightGBM")
plt.show()



# Regressão Logística
logreg_coefficients = modelos["Logistic Regression"].coef_[0]  # Coeficientes para cada variável
logreg_df = pd.DataFrame({
    'Feature': rf_feature_names,
    'Coefficient': logreg_coefficients
}).sort_values(by="Coefficient", ascending=False)


plt.figure(figsize=(8, 6))
sns.barplot(x="Coefficient", y="Feature", data=logreg_df)
plt.title("Importância das Variáveis - Logistic Regression")
plt.show()





# 8. Matrizes de Confusão para Todos os Modelos
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, (nome, modelo) in enumerate(modelos.items()):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
    axes[i].set_title(f"Matriz de Confusão - {nome}")
    axes[i].set_xlabel("Previsto")
    axes[i].set_ylabel("Real")

plt.tight_layout()
plt.show()

# Verificar a distribuição das classes
print("Distribuição das classes (Outcome):")
print(df['Outcome'].value_counts())

# Exibir a porcentagem de cada classe
print("\nPorcentagem das classes (Outcome):")
print(df['Outcome'].value_counts(normalize=True) * 100)

# Verificar distribuição das classes no treino e teste
print("\nDistribuição das classes no conjunto de TREINO:")
print(y_train.value_counts())
print("\nPorcentagem no treino:")
print(y_train.value_counts(normalize=True) * 100)

print("\nDistribuição das classes no conjunto de TESTE:")
print(y_test.value_counts())
print("\nPorcentagem no teste:")
print(y_test.value_counts(normalize=True) * 100)

