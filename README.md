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

7. Separação em features (X) e target(Y)

8. Divisão de Treino e Teste
11. Treinamento do modelo de Classificação
12. Validação Cruzada
13. Clusterin não supervisionado (K-Means)
14. Treinamento Arvore de decisão

