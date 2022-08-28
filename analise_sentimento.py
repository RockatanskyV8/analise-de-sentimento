import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk import tokenize
#  nltk.download('rslp')
import unidecode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from string import punctuation
from wordcloud import WordCloud

class Analise:

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed 
        self.stemmer = nltk.RSLPStemmer()

    def processar_frases(self, ref_column, new_column, token, palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")):
        frase_processada = list()
        for opiniao in self.dataset[ref_column]:
            nova_frase = list()
            opiniao = opiniao.lower()
            palavras_texto = token.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in palavras_irrelevantes:
                    nova_frase.append(self.stemmer.stem(palavra))
            frase_processada.append(' '.join(nova_frase))
           
        self.dataset[new_column] = frase_processada

    def classificar_texto(self, coluna_texto, coluna_classificacao):
        vetorizar = CountVectorizer(lowercase=False, max_features=50)
        bag_of_words = vetorizar.fit_transform(self.dataset[coluna_texto])
        treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                                  self.dataset[coluna_classificacao],
                                                                  random_state = self.seed)
        regressao_logistica = LogisticRegression(solver = "lbfgs")
        regressao_logistica.fit(treino, classe_treino)
        return regressao_logistica.score(teste, classe_teste)

    def nuvem_palavras_neg(self, coluna_texto):
        texto_negativo = self.dataset.query("sentiment == 'neg'")
        todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])

        nuvem_palvras = WordCloud(width= 800, height= 500,
                                  max_font_size = 110,
                                  collocations = False).generate(todas_palavras)
        plt.figure(figsize=(10,7))
        plt.imshow(nuvem_palvras, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def nuvem_palavras_pos(self, coluna_texto):
        texto_positivo = self.dataset.query("sentiment == 'pos'")
        todas_palavras = ' '.join([texto for texto in texto_positivo[coluna_texto]])

        nuvem_palvras = WordCloud(width= 800, height= 500,
                                  max_font_size = 110,
                                  collocations = False).generate(todas_palavras)
        plt.figure(figsize=(10,7))
        plt.imshow(nuvem_palvras, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def pareto(self, coluna_texto, quantidade, token):
        todas_palavras = ' '.join([texto for texto in self.dataset[coluna_texto]])
        token_frase = token.tokenize(todas_palavras)
        frequencia = nltk.FreqDist(token_frase)
        df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                      "Frequência": list(frequencia.values())})
        df_frequencia = df_frequencia.nlargest(columns = "Frequência", n = quantidade)
        plt.figure(figsize=(12,8))
        ax = sns.barplot(data = df_frequencia, x = "Palavra", y = "Frequência", color = 'gray')
        ax.set(ylabel = "Contagem")
        plt.show()

resenha = pd.read_csv("imdb-reviews-pt-br.csv")

classificacao = resenha["sentiment"].replace(["neg", "pos"], [0,1])
resenha["classificacao"] = classificacao

analise = Analise(resenha, 45)
#  teste = analise.classificar_texto("text_pt", "classificacao")
#  teste = analise.nuvem_palavras_neg("text_pt")
#  teste = analise.nuvem_palavras_pos("text_pt")
#  teste = analise.pareto("text_pt", 10)

token_espaco    = tokenize.WhitespaceTokenizer()
token_pontuacao = tokenize.WordPunctTokenizer()

analise.processar_frases("text_pt", "tratamento_1", token_espaco)

pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
pontuacao_stopwords = pontuacao + palavras_irrelevantes
analise.processar_frases("tratamento_1","tratamento_2", token_pontuacao, pontuacao_stopwords)

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

sem_acentos = [unidecode.unidecode(texto) for texto in resenha["tratamento_2"]]
resenha["tratamento_3"] = sem_acentos

analise.processar_frases("tratamento_3","tratamento_3", token_pontuacao, stopwords_sem_acento)

#  resenha["tratamento_4"] = resenha["tratamento_3"].str.lower()


print(resenha["text_pt"][0])
print(resenha["tratamento_3"][0])
#  analise.pareto("tratamento_3", 10, token_espaco)
#  print(teste)
