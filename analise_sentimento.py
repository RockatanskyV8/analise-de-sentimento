import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk import tokenize

import unidecode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from string import punctuation
from wordcloud import WordCloud

class Analise:

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed 
        self.token_pontuacao = tokenize.WordPunctTokenizer()
        self.token_espaco    = tokenize.WhitespaceTokenizer()
        #  self.stemmer         = nltk.RSLPStemmer()

    def processar_frases(self, new_column):
        palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
        frase_processada = list()
        for opiniao in self.dataset.text_pt:
            nova_frase = list()
            palavras_texto = self.token_espaco.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in palavras_irrelevantes:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))
           
        self.dataset[new_column] = frase_processada

    def processar_pontuacao(self, ref_column, new_column):
        palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")
        pontuacao = list()
        for ponto in punctuation:
            pontuacao.append(ponto)
       
        pontuacao_stopwords = pontuacao + palavras_irrelevantes
        frase_processada = list()
        for opiniao in self.dataset[ref_column]:
            nova_frase = list()
            palavras_texto = self.token_pontuacao.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in pontuacao_stopwords:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))
           
        self.dataset[new_column] = frase_processada
    
    def processar_acentuacao(self, ref_column, new_column):
        print( len(pontuacao_stopwords) )
        stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]
        sem_acentos = [unidecode.unidecode(texto) for texto in self.dataset[ref_column]]
        self.dataset[new_column] = sem_acentos

        frase_processada = list()
        for opiniao in self.dataset[new_column]:
            nova_frase = list()
            palavras_texto = self.token_pontuacao.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in stopwords_sem_acento:
                    nova_frase.append(palavra)
            frase_processada.append(' '.join(nova_frase))
           
        self.dataset[new_column] = frase_processada

    def processar_flex_deriv(self, ref_column, new_column):
        frase_processada = list()
        for opiniao in resenha[ref_column]:
            nova_frase = list()
            palavras_texto = self.token_pontuacao.tokenize(opiniao)
            for palavra in palavras_texto:
                if palavra not in stopwords_sem_acento:
                    nova_frase.append(stemmer.stem(palavra))
            frase_processada.append(' '.join(nova_frase))
           
        resenha[new_column] = frase_processada

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

    def pareto(self, coluna_texto, quantidade):
        todas_palavras = ' '.join([texto for texto in self.dataset[coluna_texto]])
        token_frase = self.token_espaco.tokenize(todas_palavras)
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
analise.processar_frases("tratamento_1")
analise.processar_pontuacao("tratamento_1","tratamento_2")
analise.processar_pontuacao("tratamento_2","tratamento_3")
teste = analise.pareto("tratamento_3", 10)
#  print(teste)
