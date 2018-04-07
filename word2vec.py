import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy
from dataset import DataSet
from scipy import spatial

dataset = DataSet()

corpus = []
#load article body into corpus
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    '''
    #stemming
    ps = PorterStemmer()
    stemmed_sentences =[]
    for w in filtered_sentence:    
        stemmed_sentences.append(ps.stem(w))
    '''
    
    corpus.append(filtered_sentence)
    
'''  
#load article headline into corpus
for s in dataset.stances:
    # raw data loading
    raw_headlines = s['Headline']
    # tokenization
    headline_tokens = word_tokenize(raw_headlines)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_headline = [w for w in headline_tokens if not w in stop_words]
    #stemming
    ps = PorterStemmer()
    stemmed_headlines =[]
    for w in filtered_headline:    
        stemmed_headlines.append(ps.stem(w))
    
    corpus.append(stemmed_headlines)
    
'''
# train word2vec

model = gensim.models.Word2Vec(corpus, size=100, window=5, min_count=5, workers=4)
print(model.wv['air'])
print(model.wv.similarity('woman', 'man'))
print(model.wv.similarity('woman', 'airport'))
model.save('word2vec')

#calculate average vector for article body
vector_article = dict.fromkeys([0])
for ID in dataset.articles:
    # raw data loading
    raw_sentences = dataset.articles[ID]
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    vector_sum = 0
    for index in range(len(filtered_sentence)):
        if filtered_sentence[index] in model.wv.vocab:
            vector_sum = vector_sum + model.wv[filtered_sentence[index]]
    average_vector = vector_sum / len(filtered_sentence)
    vector_article[ID] = average_vector
   
#calculate average vector for headline/ cosine similarity between headline and body
vector_headline = dict.fromkeys([0])
cosine_similarity = dict.fromkeys([0])
stance_index = dict.fromkeys([0])
count = 0
for s in dataset.stances:
    s['Body ID'] = int(s['Body ID'])
    count = count+1
    # raw data loading
    raw_sentences = s['Headline']
    # tokenization
    word_tokens = word_tokenize(raw_sentences)
    #stop words removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    vector_sum = 0
    for index in range(len(filtered_sentence)):
        if filtered_sentence[index] in model.wv.vocab:
            vector_sum = vector_sum + model.wv[filtered_sentence[index]]
    average_vector = vector_sum / len(filtered_sentence)
    if not np.any(average_vector):
        average_vector = [0] * 99 + [0.001]
    vector_headline[count] = average_vector
    sim = 1 - spatial.distance.cosine(average_vector, vector_article.get(s['Body ID']))
    cosine_similarity[count] = sim
    stance_index[count] = s['Stance']
    
    