## Análise de Modelos de Machine Learning para Diagnóstico de Diabetes Tipo 2

Este projeto tem como objetivo comparar diferentes algoritmos de aprendizado de máquina aplicados ao diagnóstico de diabetes tipo 2 utilizando o famoso conjunto de dados Pima Indians Diabetes Dataset.
A análise busca identificar o modelo mais eficiente em termos de acurácia, precisão, recall, F1-score e AUC.

## Contexto

O diabetes tipo 2 é uma das doenças crônicas mais prevalentes do mundo, representando um grave problema de saúde pública.
A aplicação de algoritmos de Machine Learning pode auxiliar na detecção precoce da doença a partir de variáveis clínicas simples — como glicose, pressão sanguínea, IMC e idade — reduzindo custos e permitindo intervenções preventivas.

## Tecnologias Utilizadas

Linguagem: Python 3.10+

## Bibliotecas principais

pandas — manipulação de dados

numpy — operações numéricas

matplotlib e seaborn — visualização

scikit-learn — pré-processamento, modelagem e métricas

xgboost — boosting otimizado

lightgbm — boosting leve e eficiente

imbalanced-learn — SMOTE

catboost — modelo adicional incluído no TP2

## Etapas da Análise
1. Importação e limpeza dos dados

Substituição de valores inválidos (zeros) por NaN.

Preenchimento dos NaN com a mediana da coluna.

Verificação de distribuição das classes.

2. Padronização das variáveis (TP2)

No Trabalho 2 foi incorporado:

Uso de StandardScaler para normalizar todos os atributos numéricos.
Isso melhora o desempenho de modelos baseados em gradiente e distância.

3. Redução de dimensionalidade com PCA (TP2)

Foi incluído um módulo de:

PCA (Principal Component Analysis)

Mantendo 95% da variância explicada.

O conjunto de dados passou de (8 → 7 componentes principais), reduzindo ruído e colinearidade.

4. Divisão Treino/Teste

70% treino / 30% teste

Estratificação para manter proporção original das classes

Aplicada antes do SMOTE, para evitar vazamento de dados

## Modelagem
Algoritmos utilizados (TP1 e TP2)

## Trabalho 1 — Modelos base

Regressão Logística

Random Forest

XGBoost

LightGBM

## Trabalho 2 — Melhorias e novos modelos

Modelos anteriores com tuning de hiperparâmetros

Inclusão do CatBoost

Treinamento após SMOTE + padronização + PCA

## Balanceamento das Classes (TP2)

Foi aplicada a técnica:

SMOTE — Synthetic Minority Oversampling Technique

A classe minoritária (diabéticos) foi ampliada artificialmente.

A distribuição passou de 65% / 35% para 50% / 50% no treino.

Isso elevou significativamente o recall dos modelos.

 Otimização de Hiperparâmetros (TP2)

Foi utilizado RandomizedSearchCV com validação cruzada.

Parâmetros otimizados:

## Random Forest

n_estimators

max_depth

min_samples_split, min_samples_leaf

max_features

 ## XGBoost

learning_rate

n_estimators

max_depth

subsample, colsample_bytree

 ## LightGBM

num_leaves

learning_rate

max_depth

n_estimators

## CatBoost

Ajuste leve, pois possui bom desempenho mesmo sem grande tuning.

## Avaliação dos Modelos
Métricas utilizadas:

Acurácia

Precisão

Recall

F1-score

ROC-AUC

