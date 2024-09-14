Processamento de linguagem Natural 2024.1
Any Caroliny Souza Silva (201900016796)

# **Relatório - Atividade 2**

O objetivo desta atividade foi explorar um dataset utilizando técnicas de NLP (Processamento de Linguagem Natural), realizar uma análise exploratória dos dados e treinar um modelo de classificação de texto baseado na arquitetura Transformer. Para isso, utilizamos um modelo pré-treinado disponível no Hugging Face, além de ferramentas como pandas, matplotlib, seaborn, e classes da biblioteca Transformers da Hugging Face.  

## **1. Escolha do Dataset e Modelo**  
Para a realização das tarefas, escolhemos o dataset Yahoo Answers Topics, disponível no Hugging Face. Este dataset contém perguntas, respostas e tópicos associados, sendo adequado para a tarefa de classificação de texto. O modelo pré-treinado escolhido foi o BERT-base uncased (fabriceyhc/bert-base-uncased-yahoo_answers_topics), especializado na classificação de tópicos de perguntas e respostas em inglês.  

## **2. Análise Exploratória dos Dados**  
a) Estrutura e Divisão do Dataset  
O dataset foi dividido em três partes principais:  

Treinamento  
Validação  
Teste  
Essa divisão garante que o modelo seja treinado em um conjunto de dados, validado durante o treinamento e testado em dados não vistos ao final do processo.  

b) Atributos do Dataset   
Os principais atributos do dataset incluem:  

question_title: título da pergunta  
question_content: conteúdo da pergunta  
best_answer: melhor resposta  
label: o rótulo que representa o tópico associado  
As classes de tópicos disponíveis para classificação incluem várias categorias, como "Society & Culture", "Science & Mathematics", "Computers & Internet", entre outras.  

c) Visualização dos Primeiros 5 Registros  
Para apresentar os primeiros 5 registros do dataset, utilizamos um DataFrame pandas. Cada registro inclui o título da pergunta, o conteúdo, a melhor resposta e a respectiva classe.  

```python  
import pandas as pd
df = pd.DataFrame(dataset['train']).head(5)
print(df)
```

d) Distribuição das Classes
Foi apresentada a distribuição das classes ao longo dos registros, indicando quantos exemplos cada classe contém. Utilizamos gráficos para visualizar essa distribuição:

```python  
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='label', data=dataset['train'])
plt.title('Distribuição de Classes')
plt.show()
```

e) Tamanho dos Registros por Classe
Para verificar o tamanho das perguntas por classe, criamos um gráfico de box-plot, exibindo a distribuição dos tamanhos das perguntas (em caracteres) para cada uma das classes.

```python
sns.boxplot(x='label', y=dataset['train']['question_content'].apply(len))
plt.title('Tamanho dos Registros por Classe')
plt.show()
```

## **3. Processamento com o Modelo Transformer**
f) Criação dos Tokens
Utilizamos a classe AutoTokenizer para criar os tokens a partir dos textos no dataset:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('fabriceyhc/bert-base-uncased-yahoo_answers_topics')
```

# Tokenização
```
tokenized_dataset = tokenizer(dataset['train']['question_content'], padding=True, truncation=True)
```

g) Geração dos Embeddings
Os embeddings foram gerados utilizando a classe AutoModel:

```python
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained('fabriceyhc/bert-base-uncased-yahoo_answers_topics').to(device)

inputs = torch.tensor(tokenized_dataset['input_ids']).to(device)
attention_mask = torch.tensor(tokenized_dataset['attention_mask']).to(device)

# Geração dos embeddings
with torch.no_grad():
    outputs = model(input_ids=inputs, attention_mask=attention_mask)
```

h) Extração dos Últimos Estados Escondidos  
Os últimos estados escondidos foram extraídos e apresentados para análise:  

```python
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
```

i) Conversão para Tensors PyTorch  
Todos os estados escondidos foram convertidos para tensors no PyTorch e utilizados como entrada para o modelo:  

```python
hidden_states_tensor = torch.tensor(last_hidden_states).to(device)
print(hidden_states_tensor.shape)
```

j) Estrutura do Tensor  
A estrutura do tensor foi apresentada, mostrando quantos campos possui e o número de dimensões.

k) Vetores de Treinamento e Validação  
Os vetores de treinamento e validação foram criados e seus formatos apresentados utilizando o método shape.

## **4. Visualização com UMAP**  
l) Redução Dimensional com UMAP  
Utilizamos o algoritmo UMAP para reduzir as dimensões dos embeddings e visualizar o conjunto de treinamento em 2D.

```python
import umap
embedding_umap = umap.UMAP(n_components=2).fit_transform(hidden_states_tensor.cpu().numpy())

plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=dataset['train']['label'], cmap='Spectral')
plt.title('Visualização UMAP')
plt.show()
```

m) Interpretação dos Resultados  
A análise gráfica mostrou como os diferentes tópicos (classes) estão distribuídos no espaço reduzido de duas dimensões. Observamos que algumas classes estão bem separadas, enquanto outras apresentam sobreposição, sugerindo similaridade entre os tópicos.

# **5. Classificação e Avaliação**
n) Treinamento com Regressão Logística  
Treinamos um classificador simples de regressão logística para a tarefa de classificação:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(training_vectors, training_labels)
```
A acurácia foi calculada e apresentada.  

o) Comparação com DummyClassifier  
A acurácia foi comparada com a do DummyClassifier, que fornece uma baseline para o desempenho.  

p) Matriz de Confusão  
A matriz de confusão para a regressão logística foi gerada e discutida.  

q) Treinamento com Hyperparâmetros Otimizados  
Treinamos o modelo utilizando os hyperparâmetros indicados, geramos a nova matriz de confusão e comparamos os resultados com o modelo anterior.  

## **6. Discussão dos Resultados**  
Os resultados mostraram que o modelo treinado com hyperparâmetros otimizados apresentou uma melhoria em relação à regressão logística simples, com uma matriz de confusão mais equilibrada.  