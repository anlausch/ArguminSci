import tensorflow as tf
import load_conll
import utils
import numpy as np
import os
import re
import codecs
from nltk import word_tokenize
from nltk import sent_tokenize

class Model:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def is_number(token):
	return re.match('^[\d]+[,]*.?\d*$', token) is not None


def punctuation():
	return ['—', '-', '.', ',', ';', ':', '\'', '"', '{', '}', '(', ')', '[', ']']


def assign_labels(predictions):
    string_predictions=[]
    with codecs.open("./model/config.txt", 'rb', "utf8") as file:
        for line in file.readlines():
            parts = str(line).split(": ")
            if parts[0] == "main labels":
                labels = eval(parts[1])
                break
        for i,sentence in enumerate(predictions):
            string_predictions.append([])
            for j,token in enumerate(sentence):
                for k,label in enumerate(labels):
                    if token == k:
                        string_predictions[i].append(label)
                        break
        return string_predictions


def prepare_sequence_labelling_text(texts, embeddings_vocab, pad=True, pad_token='<PAD/>', numbers_token='<NUM/>', punct_token="<PUNC/>"):
    x = []
    for i in range(len(texts)):
        if i % 2 == 0:
            print("Line: " + str(i) + " of " + str(len(texts)))
        tok_list = []
        for j in range(len(texts[i])):
            token_clean = texts[i][j]
            token = token_clean
            if token_clean.strip() in punctuation() and punct_token is not None:
                token = punct_token
            if is_number(token_clean) and numbers_token is not None:
                token = numbers_token
            if token not in embeddings_vocab and token.lower() not in embeddings_vocab:
                continue
            tok_list.append(embeddings_vocab[token] if token in embeddings_vocab else embeddings_vocab[token.lower()])
        x.append(tok_list)
    print("Line: " + str(len(texts)) + " of " + str(len(texts)))
    sequence_lengths = [len(sentence) for sentence in x]
    if pad:
        ind_pad = embeddings_vocab[pad_token]
        max_len = 1343 #max([len(t) for t in x])
        x = [t + [ind_pad] * (max_len - len(t)) for t in x]
    return np.array(x, dtype=np.int32), np.array(sequence_lengths, dtype=np.int32)



def prepare_embeddings(embd_dict, pad_token, num_token, punct_token, word2vec=False):
    embeddings_words2index = dict()
    if word2vec == True:
        embedding_matrix = np.zeros((len(embd_dict.vocab) + 3, len(embd_dict["word"])))
        for i in range(len(embd_dict.vocab)):
            embedding_vector = embd_dict[embd_dict.index2word[i]]
            embeddings_words2index[embd_dict.index2word[i]] = i
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                print("Problem with word vector")
        embedding_matrix[len(embd_dict.vocab)] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[pad_token] = len(embd_dict.vocab)

        embedding_matrix[len(embd_dict.vocab)+1] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[num_token] = len(embd_dict.vocab)+ 1

        embedding_matrix[len(embd_dict.vocab)+2] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[punct_token] = len(embd_dict.vocab) +2

        print("Embeddings are loaded and aligned")
    else:
        embedding_matrix = np.zeros((len(embd_dict.keys()) + 3, len(embd_dict["word"])))
        for i, (word, vector) in enumerate(embd_dict.items()):
            embeddings_words2index[word] = i
            if vector is not None:
                embedding_matrix[i] = vector
            else:
                print("Problem with word vector")
        embedding_matrix[len(embd_dict.keys())] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[pad_token] = len(embd_dict.keys())

        embedding_matrix[len(embd_dict.keys())+1] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[num_token] = len(embd_dict.keys()) + 1

        embedding_matrix[len(embd_dict.keys())+2] = np.random.uniform(-1.0, 1.0, len(embd_dict["word"]))
        embeddings_words2index[punct_token] = len(embd_dict.keys()) + 2

    return embedding_matrix, embeddings_words2index


def load_model():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/best-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    input_x_tensor = graph.get_tensor_by_name("input_x:0")
    dropout_keep_prob_tensor = graph.get_tensor_by_name("dropout_keep_prob:0")
    sequence_lengths_tensor = graph.get_tensor_by_name("Placeholder:0")
    arg_max = graph.get_tensor_by_name("arg_prediction/ArgMax:0")
    model = Model(input_x_tensor=input_x_tensor,
                  dropout_keep_prob_tensor=dropout_keep_prob_tensor,
                  sequence_lengths_tensor=sequence_lengths_tensor,
                  arg_max = arg_max,
                  graph = graph,
                  sess = sess)
    return model


def load_embeddings():
    if os.name == "nt":
        # embd_dict = utils.load_embeddings(
        #    "C:/Users/anlausch/workspace/cnn-text-classification/data/GoogleNews-vectors-negative300.bin", word2vec=True)
        embd_dict = utils.load_embeddings("C:/Users/anlausch/workspace/embedding_files/glove.6B/glove.6B.50d.txt",
                                          word2vec=False)
    else:
        # embd_dict = utils.load_embeddings("~/GoogleNews-vectors-negative300.bin", word2vec=True)
        embd_dict = utils.load_embeddings("./glove.6B.300d.txt", word2vec=False)
    embeddings, embedding_vocab = prepare_embeddings(embd_dict, pad_token="<PAD/>", num_token="<NUM/>",
                                                     punct_token="<PUNC/>", word2vec=False)
    return embd_dict, embedding_vocab


def predict(text, embd_vocab, model):
    text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    input, sequence_lengths = prepare_sequence_labelling_text(texts=text, embeddings_vocab=embd_vocab)

    feed_dict = {
        model.input_x_tensor: input, model.dropout_keep_prob_tensor: 1.0,
        model.sequence_lengths_tensor: sequence_lengths
    }

    labels = model.sess.run(model.arg_max, feed_dict)
    index_to_word = {v: k for k, v in embd_vocab.items()}
    labels = assign_labels(labels)
    result = [[[index_to_word[word], labels[i][j]] for j, word in enumerate(sentence) if j < sequence_lengths[i]] for i, sentence in
              enumerate(input)]
    return result


def main():
    model = load_model()
    embd_dict, embedding_vocab = load_embeddings()


    x, y_arg, y_rhet, y_aspect, y_summary, y_citation = load_conll.load_data_multiple("./annotations_conll_final_splitted/test")

    x, y_arg, classes_arg, y_rhet, classes_rhet, sequence_lengths, y_aspect, classes_aspect, y_summary, classes_summary, y_citation, classes_citation = prep_sequence_labelling(
        texts=x, labels_main=y_arg, labels_aux=y_rhet, labels_aux2=y_aspect, labels_aux3=y_summary,
        labels_aux4=y_citation, embeddings_vocab=embedding_vocab, pad=True, pad_token="<PAD/>")



    x = x[:16]
    y_arg = y_arg[:16]
    sequence_lengths = sequence_lengths[:16]

    feed_dict ={model.input_x_tensor: x, model.dropout_keep_prob_tensor: 1.0, model.sequence_lengths_tensor: sequence_lengths}

    result = model.sess.run(model.arg_max,feed_dict)
    result = assign_labels(result)
    print(result)


if __name__=='__main__':
    main()