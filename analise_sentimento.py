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

    def classificar_texto_tdif(self, coluna_texto, coluna_classificacao):
        tfidf = TfidfVectorizer(lowercase=False, max_features=50)
        tfidf_tratados = tfidf.fit_transform(self.dataset[coluna_texto])
        treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados,
                                                                      self.dataset[coluna_classificacao],
                                                                      random_state = self.seed)
        regressao_logistica = LogisticRegression(solver = "lbfgs")
        regressao_logistica.fit(treino, classe_treino)
        return regressao_logistica, tfidf.get_feature_names(), regressao_logistica.score(teste, classe_teste)

    def classificar_texto_ngrams(self, coluna_texto, coluna_classificacao):
        tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
        tfidf_tratados = tfidf.fit_transform(self.dataset[coluna_texto])
        treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados,
                                                                      self.dataset[coluna_classificacao],
                                                                      random_state = self.seed)
        regressao_logistica = LogisticRegression(solver = "lbfgs")
        regressao_logistica.fit(treino, classe_treino)
        return regressao_logistica, tfidf.get_feature_names(), regressao_logistica.score(teste, classe_teste)

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

resenha = pd.read_csv("imdb-reviews-pt-brV2.csv").drop(columns=["Unnamed: 0"], axis=1)
analise = Analise(resenha, 45)

#  reg_logistica_tdif, feature_names_tdif, tdif_score = analise.classificar_texto_tdif("tratamento_3", "classificacao")
#  print(tdif_score)

reg_logistica_ngrams, feature_names_ngrams, ngrams_score = analise.classificar_texto_ngrams("tratamento_3", "classificacao")
print(ngrams_score)

pesos = pd.DataFrame(
    reg_logistica_ngrams.coef_[0].T,
    index = feature_names_ngrams
)

print(pesos.nlargest(50,0))
print(pesos.nsmallest(10,0))

