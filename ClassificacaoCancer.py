# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Passo 1: Criar base fictícia (simulando dataset médico)
np.random.seed(42)
n_samples = 200
base = pd.DataFrame({
    'Tamanho_Tumor': np.random.uniform(0.5, 10, n_samples),  # Contínua
    'Densidade_Celular': np.random.uniform(1, 100, n_samples),  # Contínua
    'Simetria': np.random.uniform(0, 1, n_samples),  # Contínua
    'Região': np.random.choice(['Pecho', 'Pulmón', 'Próstata'], n_samples),  # Categórica
    'Tipo_Cancer': np.random.choice([0, 1], n_samples)  # 0=Benigno, 1=Maligno
})

# Passo 2: Confirmar carregamento
print("Primeiras linhas do DataFrame:\n", base.head())
print("\nShape:", base.shape)
print("\nColunas:", base.columns.tolist())

# Passo 3: Analisar e remover colunas irrelevantes
# Não há colunas irrelevantes (ex.: IDs) na base fictícia

# Passo 4: Remover valores faltantes (NaN)
print("\nValores NaN por coluna:\n", base.isnull().sum())
base = base.dropna()  # Remove linhas com NaN (não há NaN na base fictícia)

# Passo 5: Análise exploratória
# 5.1: Estatísticas descritivas
print("\nEstatísticas descritivas:\n", base.describe())

# 5.2: Boxplots para variáveis numéricas
for col in ['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']:
    plt.figure(figsize=(8, 6))
    boxplot_dict = sns.boxplot(data=base, y=col)
    plt.title(f'Boxplot de {col}')
    plt.ylabel('Valores')
    plt.grid(True)
    ax = plt.gca()
    lines = ax.get_lines()
    if len(lines) >= 5:  # Verifica se há linhas suficientes
        print(f"\nEstatísticas do boxplot de {col}:")
        print("Limite superior:", lines[4].get_ydata()[0])
        print("3º quartil:", lines[3].get_ydata()[0])
        print("Mediana:", lines[2].get_ydata()[0])
        print("1º quartil:", lines[1].get_ydata()[0])
        print("Limite inferior:", lines[0].get_ydata()[0])
    plt.show()

# 5.3: Histogramas
for col in ['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']:
    sns.histplot(base[col], kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()

# 5.4: Matriz de dispersão
fig = px.scatter_matrix(base, dimensions=['Tamanho_Tumor', 'Densidade_Celular', 'Simetria'], color='Tipo_Cancer')
fig.show()

# 5.5: Mapa de calor de correlação
base_num = base[['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']]
sns.heatmap(base_num.corr(), annot=True, cmap="coolwarm")
plt.title('Correlação entre variáveis numéricas')
plt.show()

# Passo 6: Engenharia de atributos
# 6.1: Codificação de variáveis categóricas (One-Hot Encoding para 'Região')
ohe = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(drop='first'), ['Região'])],
    remainder='passthrough'
)
base_array = ohe.fit_transform(base)
colunas_ohe = ohe.named_transformers_['OneHot'].get_feature_names_out(['Região'])
colunas_finais = list(colunas_ohe) + ['Tamanho_Tumor', 'Densidade_Celular', 'Simetria', 'Tipo_Cancer']
base = pd.DataFrame(base_array, columns=colunas_finais)
print("\nDataFrame após One-Hot Encoding:\n", base.head())

# 6.2: Padronização (escolhida) vs. Normalização (comentada)
# Padronização: Preferível aqui devido a possíveis outliers e escalas diferentes
scaler = StandardScaler()
base[['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']] = scaler.fit_transform(
    base[['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']]
)

# Normalização (alternativa, comentada)
# scaler = MinMaxScaler()
# base[['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']] = scaler.fit_transform(
#     base[['Tamanho_Tumor', 'Densidade_Celular', 'Simetria']]
# )

# Passo 7: Separação em features (X) e target (y)
X = base.drop('Tipo_Cancer', axis=1)
y = base['Tipo_Cancer']
X_entrada = X
y_saida = y

# Passo 8: Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)

# Passo 9: Treinamento da Árvore de Decisão
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_model.fit(X_train, y_train)

# Passo 10: Visualização da Árvore de Decisão
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Benigno', 'Maligno'], filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão - Diagnóstico de Câncer")
plt.show()

# Passo 11: Avaliação no teste
y_pred_dt = dt_model.predict(X_test)
print("\n=== Árvore de Decisão ===")
print("Acurácia:", accuracy_score(y_test, y_pred_dt))
print("Precisão:", precision_score(y_test, y_pred_dt, average='macro'))
print("Recall:", recall_score(y_test, y_pred_dt, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred_dt, average='macro'))

# Passo 12: Validação Cruzada
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_scores = cross_val_score(dt_model, X_entrada, y_saida, cv=kfold)
print("\nValidação Cruzada (Árvore de Decisão):")
print("Acurácia média: {:.2f}%".format(dt_scores.mean() * 100))
print("Desvio padrão: {:.2f}%".format(dt_scores.std() * 100))
print("Acurácias por fold:", dt_scores)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}
dt_validate = cross_validate(dt_model, X_entrada, y_saida, cv=kfold, scoring=scoring, return_train_score=True)
df_validate_dt = pd.DataFrame(dt_validate)
print("\nMétricas detalhadas (Árvore de Decisão):")
print(df_validate_dt)

# Passo 13: Clustering não supervisionado (K-Means)
inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inercia.append(kmeans.inertia_)
plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
base['Cluster'] = kmeans.fit_predict(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=base['Cluster'], cmap='viridis')
plt.title('Clusters em 2D (PCA)')
plt.show()
print("\nMédias por cluster:\n", base.groupby('Cluster').mean())
