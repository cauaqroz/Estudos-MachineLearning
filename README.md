# ESTUDO MACHINE LEARNING

Abaixo esta um passo a passo para implementação do projeto em machine learning, o passo a passo garante abstração para poder ser aplicado em qualquer DATASET, isso garante um entendimento do que fazer, os codigos vão ficar agrupados em um unico arquivo python posteriormente.

### Passo a Passo

1. Importar o(s) conjunto(s) de dados
    Carregue o seu arquivo em um DATAFRAME
```python
base = pd.read_csv('caminho/do/arquivo.csv',  sep=',',header=0)

# versão do colab
drive.mount('/content/drive')
base = pd.read_csv('/content/drive/MyDrive/caminho/arquivo.csv',  sep=',',header=0)
```

2. Confirmar se o DataSet foi carregado
    Confirmação rápida de que os dados carregaram corretamente e visão inicial das colunas
```python

print(base)          #Exibe DataFrame
print(base.shape)    #Exibe quantidade linhas e colunas
print(base.head())   #Exibe algumas linhas do DataFrame

# Nota! Colab nao precisa do `print` em alguns caso
```

3. Analisar e Remover colunas irrelevantes
    Remova identificadores únicos (IDs), descrições textuais longas (Nomes, Descrições, e etc..), ou atributos redundantes/sem relação com o target(coluna que voce deseja prever).
```python

base = base.drop(['Id', 'Name', 'Descrição', 'Endereço', 'NomesEspecies', 'Ticket'], axis=1)
```

4. Remover Valores faltantes(NaN)
    Liste colunas com NaN e quantifique. Depois, escolha entre:<br>
    Excluir linhas (quando poucas):

    ```python
    # consulta de valores nan na base
    base[base.isnull().any(axis=1)]

    # remover Valores NaN
    base=base.dropna()
  
    ```
    
   Imputar valores (média ou mediana para contínuos; moda para categóricos;)
   
    ```python
    # Nota! Caso seja preciso Realize essa função apos a Etapa de ANALISE EXPLORATORIAS

    ```
5. Analise exploratorias
    1. Estatísticas descritivas (Média, mediana, mínimo, máximo, quartis, contagem.)
  
    ```python
    print(base.describe())

    # Acessa as linhas que representam whiskers (limites) and quartis
    lines = ax.get_lines()

    prim_quartil   = lines[0].get_ydata()[0]
    terc_quartil   = lines[1].get_ydata()[0]
    lim_inf_premio = lines[0].get_ydata()[1]
    lim_sup_premio = lines[1].get_ydata()[1]
    mediana_premio = lines[2].get_ydata()[1]

    print("Limite superior do boxplot:", lim_sup_premio)
    print("3o quartil do boxplot:", terc_quartil)
    print("Mediana do primeiro conjunto de dados:", mediana_premio)
    print("1o quartil do boxplot:", prim_quartil)
    print("Limite inferior do boxplot:", lim_inf_premio)
    ```  
      2.  Boxplots (por variável numerica/features)<br>
     Gráfico de caixa mostra quartis e “whiskers” (possíveis outliers), ajuda a detectar valores extremos que podem distorcer médias ou treinar mal o modelo.


    ```python
    # Obter a coluna para montar o grafico
    dados = base["ColunaExemplo"]

    # Definindo as dimensões e plotando o boxplot
    plt.figure(figsize=(8, 6))

     # cria dicionário de dados
    boxplot_dict = sns.boxplot(dados)

    plt.title('Boxplot Titulo')
    plt.xlabel('Conjunto de dados')
    plt.ylabel('Valores')
    plt.grid(True)

    # Obtem eixos do gráfico ANTES DE MOSTRAR O GRÁFICO
    ax = plt.gca()

    #Obtem as linhas que represetam os limites e quartis
    lines = ax.get_lines()

    #Mostra o grafico
    plt.show()

    #Nota! aqui devemos incluir o codigo de Estatísticas descritivas visto anteriormente
    ```
    
    3. Histograma: Distribuição univariada de cada atributo numérico.
    ```python
    # Esse codigo gera um histograma para cada coluna numerica do dataFrame
    for col in ['Coluna1', 'Coluna2']:
      sns.histplot(base[col], kde=True)
      plt.title(f'Distribuição de {col}')
      plt.show()
    ```
    4. Matriz de dispersão (Scatter Matrix)
    ```python
   # Criar matriz de dispersão
    fig = px.scatter_matrix(base, dimensions=['Colunas1', 'Coluna2'],)
    fig.show()
    ```
    5. Mapa de calor de correlação: Identificar features multicolineares (que podem ser removidas ou combinadas).
     ```python
    #separar apenas atributos numéricos
    base_separada = base.loc[:, ['QuantidadeVendas','Idade','DistanciaPercorrida']]

    # Matriz de correlação
    sns.heatmap(base_separada.corr(), annot=True, cmap="coolwarm")
    plt.show()
    ```  
   
6. Engenharia de Atributos

   1. Codificação de variáveis categóricas (Label Encoding):<br> boas para variáveis ordinais, Por exemplo, para a variável `Sex` com valores `male` e `female`, pode-se atribuir `0` a `male` e `1` a `female`<br>Ideal para variáveis ordinais, ou seja, quando as categorias têm uma ordem natural (ex.: Pclass no Titanic, que representa classes de passageiro: 1ª, 2ª, 3ª, onde 1 < 2 < 3).
     ```python
    from sklearn.preprocessing import LabelEncoder

    # Codificar Sex (se necessário)
    lbl = LabelEncoder()
    base['Sex'] = lbl.fit_transform(base['Sex'])
    print(base['Sex'].head())
    ```      
   2. One-Hot Encoding:<br> para variáveis nominais, transforma uma variável categórica em colunas binárias (0 ou 1) para cada categoria. Por exemplo, para `Tipos Alimentação` (`C` e `V`), criam-se duas     colunas: `Alim_C`, `Alim_V`, onde apenas uma coluna por linha tem valor 1.<br>Ideal para variáveis nominais, onde as categorias não têm ordem natural (ex.: a alimentação não tem uma que seja mas importante que a outra).
     ```python
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Transformador para aplicar OneHot apenas na coluna "Alimentacao"
    ohe = ColumnTransformer(
        transformers=[
            ('OneHot', OneHotEncoder(drop='first'), ['Alimentacao'])
        ],
        remainder='passthrough'  # Mantém o restante das colunas
    )

    # Aplicando o transformador
    base_array = ohe.fit_transform(base)

    # Se quiser transformar em DataFrame novamente com nomes das colunas:
    colunas_ohe = ohe.named_transformers_['OneHot'].get_feature_names_out(['Alimentacao'])
    colunas_finais = list(colunas_ohe) + ['Nome', 'Idade']
    base_final = pd.DataFrame(base_array, columns=colunas_finais)

    print(base_final)
    ```

     3. Padronização:<br>Transforma os dados para que tenham média igual a 0 e desvio padrão igual a 1, usado em dados como, Idade,Preços, Classe, que podem causar interferencia no modelo, A padronização equaliza a influência de todas as variáveis.
     ```python
   from sklearn.preprocessing import StandardScaler

    #Quando usar: Para algoritmos como MLP, SVM, KNN, K-Means
    #Quando as variáveis têm escalas diferentes (ex.: Age vs. Pclass no Titanic).
    #Quando há outliers (ex.: valores altos de Age ou Fare na base completa do Titanic).
     
    scaler = StandardScaler()
     
    base[['Pclass', 'Sex', 'Age']] = scaler.fit_transform(base[['Pclass', 'Sex', 'Age']])
    ```

     4. Normalização:<br>Transforma os dados para um intervalo fixo, geralmente [0, 1], ou outro intervalo especificado (ex.: [-1, 1]), usada quando há intervalos especificados, como em Grau de Felicidade, que vai de 0-10, a normalização garante que todas variaveis contribuam igualmente para as distancias de modelos como KNN e K-means
     ```python
    from sklearn.preprocessing import MinMaxScaler

    #Quando usar: Para algoritmos baseados em distancia como KNN, K-Means
    #Quando os dados não têm outliers significativos, pois a normalização é sensível a valores extremos.
    #Quando as variáveis já estão em escalas semelhantes(ex.: 0-10), mas precisam de um intervalo fixo (ex.: [0, 1]).
     

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(base[['Pclass', 'Sex', 'Age']])

    base['Grau de Felicidade Normalizado'] = scaler.fit_transform(base[['Grau de Felicidade']])
    ```
        
7. Separação em features (X) e target(Y): Isole a coluna alvo (classificação ou regressão) como vetor y, e todas as demais como matriz X.
```python
# Separar entrada (X) e saída (Y)
X = base.drop('Col_Alvo', axis=1)
Y = base['Col_Alvo']

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

8. Divisão de Treino e Teste: Dividir os dados em conjunto de treino (ex.: 70%) e teste (ex.: 30%)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)
```

9. Treinamento do modelo de Classificação

   1.MLP-Classifier: projetado para problemas de classificação, Prevê rótulos categóricos (ex.: classes 0/1, "sim/não", ou múltiplas classes).
   
    ```python
    # Treinar com fit(X_train, y_train).
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='logistic', solver='adam', random_state=42, verbose=True)
    mlp.fit(X_train, Y_train)

    # Verificar Acurácia Inicial
    # Underfitting: Acurácia baixa em treino e teste → aumentar camadas/neurônios ou iterações.
    # Overfitting: Acurácia alta em treino, baixa em teste → reduzir camadas/neurônios, adicionar regularização

    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))

    print("Acurácia do treinamento:", train_acc)
    print("Acurácia do treino:", test_acc)

    from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
    y_pred = mlp.predict(X_test)
    print("Precisão:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1-Score:", f1_score(y_test, y_pred, average='macro'))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    ```
    
    2.MLP-Regressor: rede neural multicamadas para problemas de regressão, Prevê valores contínuos (ex.: preço, temperatura, pontuação), Usa funções de perda como erro quadrático médio (MSE) para otimizar a diferença entre previsões e valores reais.
   
    ```python
    # Passo 5: Treinamento do MLPRegressor
    start_time = time()
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, activation='logistic', solver='adam', random_state=42, verbose=True)
    mlp.fit(X_train, y_train)
    fit_time = time() - start_time
    print("Tempo de treinamento (MLPRegressor):", fit_time)

    # Avaliação no treino
    y_pred_train = mlp.predict(X_train)
    print("\nMétricas no treinamento (MLPRegressor):")
    print("MSE:", mean_squared_error(y_train, y_pred_train))
    print("MAE:", mean_absolute_error(y_train, y_pred_train))
    print("RMSE:", mean_squared_error(y_train, y_pred_train, squared=False))
    print("R²:", r2_score(y_train, y_pred_train))
    
    # Avaliação no teste
    y_pred_test = mlp.predict(X_test)
    print("\nMétricas no teste (MLPRegressor):")
    print("MSE:", mean_squared_error(y_test, y_pred_test))
    print("MAE:", mean_absolute_error(y_test, y_pred_test))
    print("RMSE:", mean_squared_error(y_test, y_pred_test, squared=False))
    print("R²:", r2_score(y_test, y_pred_test))
    ```
    
10. Validação Cruzada: dividir treino em k folds (ex.: 5-fold), treinar e testar em cada dobra, e coletar métricas.
```Python
#definir a quantidade de dobras
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#realizar a validação cruzada
scores = cross_val_score(mlp, X_entrada, y_saida, cv=kfold)
print("Acurácia média: {:.2f}%".format(scores.mean() * 100))
print("Desvio padrão: {:.2f}%".format(scores.std() * 100))
print("Acurácia: ",scores)
```
        
11. Clusterin não supervisionado (K-Means): Treinar KMeans para diferentes números de clusters (ex.: 1 a 10) e plotar a inércia (soma dos quadrados das distâncias aos centroides) contra k. O "cotovelo" indica o número ótimo de clusters.

```Python
from sklearn.cluster import KMeans
inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, random_state=42)
base['Cluster'] = kmeans.fit_predict(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=base['Cluster'], cmap='viridis')
plt.title('Clusters em 2D (PCA)')
plt.show()

print(base.groupby('Cluster').mean())
```
    
12. Treinamento Arvore de decisão: RandomForestClassifier é um modelo de ensemble que combina várias árvores de decisão para melhorar a precisão e reduzir o overfitting.<br>
- X_train, X_test, y_train, y_test são os dados de treino e teste já preparados.<br>
- X_entrada e y_saida são os dados completos (antes da divisão treino/teste).<br>
- kfold é um objeto de validação cruzada (ex.: StratifiedKFold).<br>
- base é o DataFrame com os dados originais.

```Python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy")
rf_model.fit(X_train, y_train)

#Previsão com o Modelo
rf_preds = rf_model.predict(X_test)

# Avaliação
print("=== Random Forest ===")
print("Acurácia:", accuracy_score(y_test, rf_preds))
```

- Visualização da Arvore de Decisão
```Python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

arvore = rf_model.estimators_[0]

# Plot the decision tree
plt.figure(figsize=(20, 10))
colunas = base.columns[1:5].tolist()
plot_tree(arvore, feature_names=colunas, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()

#Validação cruzada com acuracia
rf_scores = cross_val_score(rf_model, X_entrada, y_saida, cv=kfold)
print("Acurácia média: {:.2f}%".format(rf_scores.mean() * 100))
print("Desvio padrão: {:.2f}%".format(rf_scores.std() * 100))
print("Acurácia: ", rf_scores)

#Validação Cruzada com Múltiplas Métricas
precisao = make_scorer(precision_score, average='macro')
recall = make_scorer(recall_score, average='macro')
rf_validate = cross_validate(rf_model, X_entrada, y_saida,
  cv=kfold, scoring={'accuracy': 'accuracy', 'precision': precisao, 'recall': recall},
  return_train_score=True)
df_validate = pd.DataFrame(rf_validate)
df_validate
```
