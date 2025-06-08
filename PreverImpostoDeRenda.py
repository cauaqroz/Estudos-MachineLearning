# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from time import time

# Passo 1: Criar base fictícia (simulando dataset financeiro)
np.random.seed(42)
n_samples = 200
base = pd.DataFrame({
    'Renda_Anual': np.random.uniform(20000, 150000, n_samples),  # Contínua
    'Idade': np.random.randint(18, 80, n_samples),  # Contínua
    'Tipo_Emprego': np.random.choice(['Asalariado', 'Autônomo', 'Empresário'], n_samples),  # Categórica
    'Imposto_Renda': np.random.uniform(1000, 30000, n_samples)  # Contínuo
})

# Passo 2: Confirmar carregamento
print("Primeiras linhas do DataFrame:\n", base.head())
print("\nShape:", base.shape)
print("\nColunas:", base.columns.tolist())

# Passo 3: Analisar e remover colunas irrelevantes
# Não há colunas irrelevantes na base fictícia

# Passo 4: Remover valores faltantes (NaN)
print("\nValores NaN por coluna:\n", base.isnull().sum())
base = base.dropna()  # Remove linhas com NaN (não há NaN na base fictícia)

# Passo 5: Análise exploratória
# 5.1: Estatísticas descritivas
print("\nEstatísticas descritivas:\n", base.describe())

# 5.2: Boxplots para variáveis numéricas
for col in ['Renda_Anual', 'Idade', 'Imposto_Renda']:
    plt.figure(figsize=(8, 6))
    boxplot_dict = sns.boxplot(data=base, y=col)
    plt.title(f'Boxplot de {col}')
    plt.ylabel('Valores')
    plt.grid(True)
    ax = plt.gca()
    lines = ax.get_lines()
    if len(lines) >= 5:
        print(f"\nEstatísticas do boxplot de {col}:")
        print("Limite superior:", lines[4].get_ydata()[0])
        print("3º quartil:", lines[3].get_ydata()[0])
        print("Mediana:", lines[2].get_ydata()[0])
        print("1º quartil:", lines[1].get_ydata()[0])
        print("Limite inferior:", lines[0].get_ydata()[0])
    plt.show()

# 5.3: Histogramas
for col in ['Renda_Anual', 'Idade', 'Imposto_Renda']:
    sns.histplot(base[col], kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()

# 5.4: Matriz de dispersão
fig = px.scatter_matrix(base, dimensions=['Renda_Anual', 'Idade', 'Imposto_Renda'])
fig.show()

# 5.5: Mapa de calor de correlação
base_num = base[['Renda_Anual', 'Idade', 'Imposto_Renda']]
sns.heatmap(base_num.corr(), annot=True, cmap="coolwarm")
plt.title('Correlação entre variáveis numéricas')
plt.show()

# Passo 6: Engenharia de atributos
# 6.1: One-Hot Encoding para 'Tipo_Emprego'
ohe = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(drop='first'), ['Tipo_Emprego'])],
    remainder='passthrough'
)
base_array = ohe.fit_transform(base)
colunas_ohe = ohe.named_transformers_['OneHot'].get_feature_names_out(['Tipo_Emprego'])
colunas_finais = list(colunas_ohe) + ['Renda_Anual', 'Idade', 'Imposto_Renda']
base = pd.DataFrame(base_array, columns=colunas_finais)
print("\nDataFrame após One-Hot Encoding:\n", base.head())

# 6.2: Normalização (escolhida) vs. Padronização (comentada)
# Normalização: Preferível devido à ausência de outliers significativos e intervalo desejado
scaler = MinMaxScaler()
base[['Renda_Anual', 'Idade']] = scaler.fit_transform(base[['Renda_Anual', 'Idade']])

# Padronização (alternativa, comentada)
# scaler = StandardScaler()
# base[['Renda_Anual', 'Idade']] = scaler.fit_transform(base[['Renda_Anual', 'Idade']])

# Passo 7: Separação em features (X) e target (y)
X = base.drop('Imposto_Renda', axis=1)
y = base['Imposto_Renda']
X_entrada = X
y_saida = y

# Passo 8: Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

# Passo 9: Treinamento da Árvore de Decisão
start_time = time()
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
print("Tempo de treinamento (DecisionTreeRegressor):", time() - start_time)

# Passo 10: Visualização da Árvore de Decisão
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão - Previsão de Imposto de Renda")
plt.show()

# Passo 11: Avaliação no teste
y_pred_dt = dt_model.predict(X_test)
print("\n=== Árvore de Decisão ===")
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("RMSE:", mean_squared_error(y_test, y_pred_dt, squared=False))
print("R²:", r2_score(y_test, y_pred_dt))

# Passo 12: Validação Cruzada
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score)
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
