# Atividade 2 – Classificação de Texto


1) Escolha um dos datasets indicados nas URLs acima. 

2) Escolha um dos modelos disponíveis no Hugging Face utilizando os filtros Natural Language Processing “text classification” e Language “<idioma do dataset escolhido por você>” para que este modelo seja utilizado nesta atividade.Uma curiosidade em relação à arquitetura de modelos Transformers (que são a base para os LLMs) é que modelos especializados em classificação do texto são em sua maioria do tipo encoder-only. Como você poderia justificar isto?

3) Utilize o modelo e o dataset selecionados para executar as seguintes atividades via programação utilizando sempre recursos da linguagem Python e tendo como referência o notebook do Capítulo 2 do livro indicado acima:

a) Definição e apresentação da estrutura e divisão do dataset para treinamento, validação e teste.

b) Apresentação dos atributos que compõem o dataset, incluindo as opções de classes a serem fornecidas como resultado da classificação.

c) Apresentação dos primeiros cinco registros que compõem o dataset com os respectivos campos e classe(s) associada(s) através de um dataframe.

d) Apresentação da distribuição das classes ao longo dos registros, incluindo um gráfico com o quantitativo de registros por classe.

e) Apresentação do tamanho dos registros por cada classe usando um gráfico de box-plots.

f) Utilize a classe AutoTokenizer do Transformer através do modelo pré-treinado escolhido por você para a criação dos tokens do dataset.

g) Utilize o modelo pré-treinado escolhido por você através da classe AutoModel do Transformer para gerar os embeddings resultantes dos estados escondidos.

h) Extraia e apresente os últimos estados escondidos.

i) Converta todos os estados escondidos para uma estrutura tensor do PyTorch e passe-os como entrada para o modelo escolhido.

j) Apresente a estrutura do tensor, incluindo quantos campos possui e o número de dimensões utilizados para representá-lo.

k) Crie os vetores de treinamento e validação e apresente o seu formato utilizando o método shape.

l) Apresente a visualização do conjunto de treinamento utilizando o algoritmo UMAP para reduzir o número de dimensões existentes para duas dimensões. Apresente a visualização de forma similar à figura apresentada na página 43 do capítulo 2.

m) Interprete os resultados apresentados em relação às diferenças representadas graficamente entre cada classe apresentada no diagrama.

n) Treine os dados com um classificador simples de regressão logística e apresente o resultado de acurácia correspondente para a atividade de classificação de texto.

o) Compare com o resultado de acurácia do DummyClassifier.

p) Apresente a matriz de confusão do uso da regressão logística e discuta os resultados obtidos.

q) Execute o treinamento do modelo usando os hyperparametros indicados na páǵina 48 e apresente a matriz de confusão correspondente.

r) Discuta e compare os novos resultados obtidos com os resultados apresentados anteriormente com a regressão logística.