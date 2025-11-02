#  Análise de Modelos de Machine Learning para Diagnóstico de Diabetes Tipo 2

Este projeto tem como objetivo **comparar diferentes algoritmos de aprendizado de máquina** aplicados ao diagnóstico de **diabetes tipo 2** utilizando o famoso conjunto de dados **Pima Indians Diabetes Dataset**.  
A análise busca identificar o modelo mais eficiente em termos de acurácia, precisão, recall, F1-score e AUC.

---

##  Contexto

O diabetes tipo 2 é uma das doenças crônicas mais prevalentes do mundo, representando um grave problema de saúde pública.  
A aplicação de algoritmos de *Machine Learning* pode auxiliar na **detecção precoce** da doença a partir de variáveis clínicas simples — como glicose, pressão sanguínea, IMC e idade — reduzindo custos e permitindo intervenções preventivas.

---

##  Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Bibliotecas principais:**
  - `pandas` — manipulação e análise de dados
  - `numpy` — cálculos numéricos
  - `matplotlib` e `seaborn` — visualização de dados
  - `scikit-learn` — modelagem e métricas de desempenho
  - `xgboost` — algoritmo de boosting otimizado
  - `lightgbm` — boosting leve e eficiente desenvolvido pela Microsoft

## Etapas da Análise

Importação e limpeza dos dados

Substituição de valores inválidos (zeros) por NaN e preenchimento com a mediana.

Padronização das variáveis numéricas.

Divisão treino/teste

70% para treino e 30% para teste (com estratificação para manter proporção entre classes).

## Modelagem
 
 Algoritmos utilizados:

Regressão Logística

Random Forest

XGBoost

LightGBM

## Avaliação

 Métricas: Acurácia, Precisão, Recall, F1-score e ROC-AUC

 Visualizações:

Importância das variáveis em cada modelo

Matrizes de confusão para comparação dos acertos e erros


