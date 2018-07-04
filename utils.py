from nltk.corpus import stopwords
import re
import numpy as np
import math
import os
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import nltk
import codecs
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import regex


def remove_stop_words_from_sentence_list(texts):
    english_stopwords = stopwords.words("english")
    pattern = r'\b(' + r'|'.join(english_stopwords) + r')\b\s*'
    return [re.sub(pattern,'', text, flags=re.IGNORECASE) for text in texts]

'''Pattern explained:
\( citation markers start with parentheses
then the author group starts
[A-Z] author names start with a capital letter
[a-z]+ followed by at least one lowercase letter
then there is a whitespace except if there are no other authors
if there are other authors we habe (and) or (et al.)
there can be more than one author
before the year, there is a comma and a whitespace
(18|19|20)\d\d then a reasonable date
optionally a second citation follows, if yes, we need a semicolon and a soace (; )?
# )+ for multiple citations
#\) closing parantheses
'''
def remove_citation_marker(s):
    #pattern = r'\(((([A-Z][a-z]+)( )*(and)*( )*)+(et al.)*, (18|19|20)\d\d(; )?)+\)'
    pattern = regex.compile(r'\(?((([A-Z][a-z]+)( )*(and)*( )*)+(et al.)*,? \(?(18|19|20)\d\d\)?(;*,* )?)+\)?')
    s = regex.sub(pattern, '', s)
    return s


def remove_stop_words_from_single_sentence(text):
    english_stopwords = stopwords.words("english")
    pattern = r'\b(' + r'|'.join(english_stopwords) + r')\b\s*'
    return re.sub(pattern,'', text, flags=re.IGNORECASE)


def compute_standard_error(prediction):
    prediction = list(map(int, prediction))
    return float(np.std(prediction)) / math.sqrt(len(prediction))


def get_write_mode(path):
    if os.path.exists(path):
        return 'a'  # append if already exists
    else:
        return 'w'  # make a new file if not


def shuffle_data(x, y):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    return x[shuffle_indices], y[shuffle_indices]


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# returns a dictionary of embeddings
def load_embeddings(path, word2vec=False, rdf2vec=False):
    embbedding_dict = {}
    if word2vec == False and rdf2vec == False:
        with codecs.open(path, "rb", "utf8", "ignore") as infile:
            for line in infile:
                parts = line.split()
                word = parts[0]
                nums = [float(p) for p in parts[1:]]
                embbedding_dict[word] = nums
        return embbedding_dict
    elif word2vec == True:
        #Load Google's pre-trained Word2Vec model.
        if os.name != 'nt':
            model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        else:
            model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        return model
    elif rdf2vec == True:
        #Load Petars model.
        model = gensim.models.Word2Vec.load(path)
        return model


class MeanEmbeddingVectorizer(object):
    def __init__(self, embds):
        self.embds = embds
        #self.dim = next(iter(embds))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #return np.array([np.mean([self.embds[w] for w in words if w in self.embds]
        #                            or [np.zeros(self.dim)], axis=0) for words in X])
        sentence_vectors = [np.mean([self.embds[word] for word in nltk.word_tokenize(sentence) if word in self.embds], axis=0) for sentence in X]
        for i, sentence_vector in enumerate(sentence_vectors):
            if type(sentence_vector) is not np.ndarray:
                sentence_vectors[i] = np.array([0.0 for j in range(0, len(sentence_vectors[0]))], dtype=float)
        return sentence_vectors


class MeanEmbeddingVectorizerSingleSentence(object):
    def __init__(self, embds):
        self.embds = embds
        #self.dim = next(iter(embds))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #return np.array([np.mean([self.embds[w] for w in words if w in self.embds]
        #                            or [np.zeros(self.dim)], axis=0) for words in X])
        sentence_vector = np.mean([self.embds[word] for word in nltk.word_tokenize(X) if word in self.embds], axis=0)
        return sentence_vector


class MeanDBpediaEmbeddingVectorizerSingleSentence(object):
    def __init__(self, embds):
        self.embds = embds
        #self.dim = next(iter(embds))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #return np.array([np.mean([self.embds[w] for w in words if w in self.embds]
        #                            or [np.zeros(self.dim)], axis=0) for words in X])
        dbr_tokens = ["dbr:" + str(word) for word in nltk.word_tokenize(X)]
        sentence_vector = np.mean([self.embds[word] for word in dbr_tokens if word in self.embds], axis=0)
        if not isinstance(sentence_vector, np.float64):
            sentence_vector = np.array([number if not math.isnan(number) else 0 for number in sentence_vector])
        else:
            np.array(sentence_vector)
        return sentence_vector


class TfidfEmbeddingVectorizer(object):
    def __init__(self, embds, tfidf_vectorizer):
        self.embds = embds
        self.tfidf_vectorizer = tfidf_vectorizer
        max_idf = max(self.tfidf_vectorizer.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w,  self.tfidf_vectorizer.idf_[i]) for w, i in  self.tfidf_vectorizer.vocabulary_.items()])

    def transform(self, X):
        sentence_vectors = []
        for x in X:
            tokens = nltk.word_tokenize(x)
            sentence_vector = np.mean([[value * self.word2weight[word] * tokens.count(word) for value in self.embds[word]] for word in tokens if word in self.embds], axis=0)
            sentence_vectors.append(sentence_vector)
        return sentence_vectors


class TfidfEmbeddingVectorizerSingleSentence(object):
    def __init__(self, embds, tfidf_vectorizer):
        self.embds = embds
        self.word2weight = None
        self.tfidf_vectorizer = tfidf_vectorizer
        max_idf = max(self.tfidf_vectorizer.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w,  self.tfidf_vectorizer.idf_[i]) for w, i in  self.tfidf_vectorizer.vocabulary_.items()])

    def transform(self, X):
        tokens = nltk.word_tokenize(X)
        sentence_vector = np.mean([[value * self.word2weight[word] * tokens.count(word) for value in self.embds[word]] for word in tokens if word in self.embds], axis=0)
        return sentence_vector


'''
performs removal of citation markers, removal of punctuation, lemmatization and stopword removal for a given string s
'''
def preprocess_string_tfidf(s):
    #s = remove_citation_marker(s)
    #s = re.sub(r'[^\w\s]', ' ', s).strip()
    try:
        tokenized_string = [token for token in word_tokenize(s.lower())]
    except Exception as e:
        print(e)
        print(word_tokenize(s.lower))
    return ' '.join(tokenized_string)

'''
performs removal of citation markers, removal of punctuation, lemmatization and stopword removal for a given string s
'''
def preprocess_string(s):
    #s = remove_citation_marker(s)
    s = re.sub(r'[^\w\s]', ' ', s).strip()
    wordnet_lemmatizer = WordNetLemmatizer()
    #stemmer = PortStemmer()
    english_stopwords = stopwords.words("english")
    tokenized_string = [wordnet_lemmatizer.lemmatize(token) for token in word_tokenize(s.lower()) if token not in english_stopwords]
    return ' '.join(tokenized_string)

'''
performs removal of citation markers, removal of punctuation, lemmatization and stopword removal for a given list of strings
'''
def preprocess_string_list(s_list):
    return [preprocess_string(s) for s in s_list]

def preprocess_string_list_tfidf(s_list):
    return [preprocess_string_tfidf(s) for s in s_list]


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def align_vocab_and_embd(embd_dict, vocab):
    embd_list = []
    empty_list = []
    for i,token in enumerate(vocab):
        if token in embd_dict:
            float_embd = np.array([float(x) for x in embd_dict[token]])
            embd_list.append(float_embd)
        else:
            empty_embd = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
            embd_list.append(empty_embd)
            empty_list.append(i)
    return np.array(embd_list), np.array(empty_list)