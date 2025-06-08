import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree
from time import time

# --- Passo 1: Importar o conjunto de dados ---
# Carregar o dataset Titanic (assumindo que está no Google Drive)
from google.colab import drive
drive.mount('/content/drive')
base = pd.read_csv('/content/drive/MyDrive/titanic.csv', sep=',', header=0)

# --- Passo 2: Confirmar se o dataset foi carregado ---
print("DataFrame:")
print(base)
print("\nDimensões (linhas, colunas):", base.shape)
print("\nPrimeiras linhas:")
print(base.head())

# --- Passo 3: Remover colunas irrelevantes ---
# Remover colunas especificadas no resumo
base = base.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Embarked'], axis=1)
print("\nColunas após remoção:", base.columns.tolist())

# --- Passo 4: Remover ou imputar valores faltantes (NaN) ---
# Verificar valores NaN
print("\nLinhas com valores NaN:")
print(base[base.isnull().any(axis=1)])

# Imputar valores faltantes em 'Age' com a mediana (mais robusto a outliers)
base['Age'] = base['Age'].fillna(base['Age'].median())
base['Cabin'] = base['Cabin'].fillna('Unknown')  # Preencher Cabin com 'Unknown'

# Verificar se ainda há NaN
print("\nValores NaN após imputação:", base.isnull().sum().sum())

# --- Passo 5: Análise exploratória ---
# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(base.describe())

# Boxplot para 'Age'
plt.figure(figsize=(8, 6))
boxplot_dict = sns.boxplot(data=base['Age'])
plt.title('Boxplot de Idade')
plt.xlabel('Conjunto de dados')
plt.ylabel('Idade')
plt.grid(True)
ax = plt.gca()
lines = ax.get_lines()
prim_quartil = lines[0].get_ydata()[0]
terc_quartil = lines[1].get_ydata()[0]
lim_inf_age = lines[0].get_ydata()[1]
lim_sup_age = lines[1].get_ydata()[1]
mediana_age = lines[2].get_ydata()[1]
print("\nBoxplot de Idade:")
print("Limite superior:", lim_sup_age)
print("3º quartil:", terc_quartil)
print("Mediana:", mediana_age)
print("1º quartil:", prim_quartil)
print("Limite inferior:", lim_inf_age)
plt.show()

# Histograma para 'Age' e 'Pclass'
for col in ['Age', 'Pclass']:
    plt.figure(figsize=(8, 6))
    sns.histplot(base[col], kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()

# Matriz de dispersão
fig = px.scatter_matrix(base, dimensions=['Pclass', 'Age'])
fig.show()

# Mapa de calor de correlação (apenas numéricas)
base_num = base[['Pclass', 'Age', 'Survived']]
plt.figure(figsize=(8, 6))
sns.heatmap(base_num.corr(), annot=True, cmap="coolwarm")
plt.title('Mapa de Calor de Correlação')
plt.show()

# --- Passo 6: Engenharia de atributos ---
# Codificação de variáveis categóricas
# Label Encoding para 'Sex' (male=0, female=1)
lbl = LabelEncoder()
base['Sex'] = lbl.fit_transform(base['Sex'])
print("\nPrimeiras linhas de 'Sex' após Label Encoding:")
print(base['Sex'].head())

# One-Hot Encoding para 'Cabin' (considerando apenas as primeiras letras das cabines)
base['Cabin'] = base['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
ohe = ColumnTransformer(
    transformers=[
        ('OneHot', OneHotEncoder(drop='first'), ['Cabin'])
    ],
    remainder='passthrough'
)
base_array = ohe.fit_transform(base)
colunas_ohe = ohe.named_transformers_['OneHot'].get_feature_names_out(['Cabin'])
colunas_finais = list(colunas_ohe) + ['Pclass', 'Sex', 'Age', 'Survived']
base = pd.DataFrame(base_array, columns=colunas_finais)
print("\nDataFrame após One-Hot Encoding:")
print(base.head())

# Padronização de 'Pclass' e 'Age'
scaler = StandardScaler()
base[['Pclass', 'Age']] = scaler.fit_transform(base[['Pclass', 'Age']])
print("\nDataFrame após padronização:")
print(base.head())

# Normalização alternativa (exemplo com MinMaxScaler)
scaler_norm = MinMaxScaler()
base['Age_Normalizado'] = scaler_norm.fit_transform(base[['Age']])

# --- Passo 7: Separação em features (X) e target (y) ---
X = base.drop('Survived', axis=1)
y = base['Survived']
X_scaled = scaler.fit_transform(X)  # Padronizar todas as features

# --- Passo 8: Divisão de treino e teste ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)
print("\nDimensões X_train:", X_train.shape)
print("Dimensões X_test:", X_test.shape)

# --- Passo 9: Treinamento do modelo de classificação ---
# MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='logistic', solver='adam', random_state=42, verbose=True)
start_time = time()
mlp.fit(X_train, y_train)
fit_time = time() - start_time
print("\nTempo de treinamento (MLPClassifier):", fit_time)

# Avaliação MLPClassifier
y_pred_mlp = mlp.predict(X_test)
train_acc = accuracy_score(y_train, mlp.predict(X_train))
test_acc = accuracy_score(y_test, y_pred_mlp)
print("\nMétricas MLPClassifier:")
print("Acurácia no treino:", train_acc)
print("Acurácia no teste:", test_acc)
print("Precisão:", precision_score(y_test, y_pred_mlp, average='macro'))
print("Recall:", recall_score(y_test, y_pred_mlp, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred_mlp, average='macro'))
print("MSE:", mean_squared_error(y_test, y_pred_mlp))
print("MAE:", mean_absolute_error(y_test, y_pred_mlp))
print("RMSE:", mean_squared_error(y_test, y_pred_mlp, squared=False))

# RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("\n=== Random Forest ===")
print("Acurácia:", accuracy_score(y_test, rf_preds))

# Visualização de uma árvore do Random Forest
plt.figure(figsize=(20, 10))
arvore = rf_model.estimators_[0]
plot_tree(arvore, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão do Random Forest")
plt.show()

# --- Passo 10: Validação cruzada ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Validação cruzada para MLPClassifier
mlp_scores = cross_val_score(mlp, X_scaled, y, cv=kfold)
print("\nValidação Cruzada (MLPClassifier):")
print("Acurácia média: {:.2f}%".format(mlp_scores.mean() * 100))
print("Desvio padrão: {:.2f}%".format(mlp_scores.std() * 100))
print("Acurácias:", mlp_scores)

# Validação cruzada para RandomForestClassifier com múltiplas métricas
precisao = make_scorer(precision_score, average='macro')
recall = make_scorer(recall_score, average='macro')
rf_validate = cross_validate(rf_model, X_scaled, y, cv=kfold,
                             scoring={'accuracy': 'accuracy', 'precision': precisao, 'recall': recall},
                             return_train_score=True)
df_validate = pd.DataFrame(rf_validate)
print("\nValidação Cruzada (RandomForestClassifier):")
print(df_validate)

# --- Passo 11: Clustering não supervisionado (K-Means) ---
inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# Aplicar K-Means com k=3 (baseado no cotovelo)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=42)
base['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=base['Cluster'], cmap='viridis')
plt.title('Clusters em 2D (PCA)')
plt.show()

print("\nMédias por cluster:")
print(base.groupby('Cluster').mean())
