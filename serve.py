import tensorflow as tf
import numpy as np
import os
import re
import codecs
from nltk import word_tokenize
from nltk import sent_tokenize
import pickle

class Model:
    def __init__(self, type):
        self.type = type
        self.path = os.path.join("./model", type)
        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(self.path + '/best-model.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path))
            self.input_tensor = self.graph.get_tensor_by_name('input_x:0')
            self.dropout_keep_prob_tensor = self.graph.get_tensor_by_name('dropout_keep_prob:0')
            self.sequence_lengths_tensor = self.graph.get_tensor_by_name('Placeholder:0')
            if type == "argumentation":
                self.arg_max = self.graph.get_tensor_by_name("arg_prediction/ArgMax:0")
            elif type == "discourse":
                self.arg_max = self.graph.get_tensor_by_name("auxiliary_prediction/ArgMax:0")

            self.embedding_dict, self.embedding_vocab = self.load_embeddings(self.path)


    def is_number(self, token):
        return re.match('^[\d]+[,]*.?\d*$', token) is not None


    def punctuation(self):
        return ['—', '-', '.', ',', ';', ':', '\'', '"', '{', '}', '(', ')', '[', ']']


    def begin_of_span(self):
        return ["Token_Label.BEGIN_BACKGROUND_CLAIM", "Token_Label.BEGIN_OWN_CLAIM", "Token_Label.BEGIN_DATA"]


    def assign_labels(self, predictions):
        string_predictions=[]
        with codecs.open(os.path.join(self.path, "config.txt"), 'rb', "utf8") as file:
            for line in file.readlines():
                parts = str(line).split(": ")
                if parts[0] == "main labels":
                    labels = eval(parts[1])
                    break
            if self.type == "argumentation":
                for i,sentence in enumerate(predictions):
                    string_predictions.append([])
                    for j,token in enumerate(sentence):
                        for k,label in enumerate(labels):
                            if token == k:
                                string_predictions[i].append(label)
                                break
            else:
                for i,sentence in enumerate(predictions):
                    string_predictions.append([])
                    for k, label in enumerate(labels):
                        if sentence == k:
                            string_predictions[i].append(label)
                            break
            return string_predictions


    def prepare_sequence_labelling_text(self, texts, pad=True, pad_token='<PAD/>', numbers_token='<NUM/>', punct_token="<PUNC/>"):
        x = []
        punctuation_list = []
        numbers_list = []
        removed_tokens = {}
        for i in range(len(texts)):
            if i % 2 == 0:
                print("Line: " + str(i) + " of " + str(len(texts)))
            tok_list = []
            for j in range(len(texts[i])):
                token_clean = texts[i][j]
                token = token_clean
                if token_clean.strip() in self.punctuation() and punct_token is not None:
                    punctuation_list.append(token)
                    token = punct_token
                if self.is_number(token_clean) and numbers_token is not None:
                    numbers_list.append(token)
                    token = numbers_token
                if token not in self.embedding_vocab and token.lower() not in self.embedding_vocab:
                    removed_tokens[str(i) + "_" + str(j)] = token
                    continue
                tok_list.append(self.embedding_vocab[token] if token in self.embedding_vocab else self.embedding_vocab[token.lower()])
            x.append(tok_list)
        print("Line: " + str(len(texts)) + " of " + str(len(texts)))
        sequence_lengths = [len(sentence) for sentence in x]
        if pad:
            ind_pad = self.embedding_vocab[pad_token]
            if self.type == "argumentation":
                max_len = 167 #max([len(t) for t in x])
            else:
                max_len = 1343
            x = [t + [ind_pad] * (max_len - len(t)) for t in x]
            real_length = len(x)
            while len(x) % 16 != 0:
                x.append([ind_pad for i in range(max_len)])
                sequence_lengths.append(0)

        return np.array(x, dtype=np.int32), np.array(sequence_lengths, dtype=np.int32), np.array(punctuation_list), \
               np.array(numbers_list), removed_tokens, real_length


    def load_embeddings(self, path):
        embedding_vocab = pickle.load(open(path + "/embedding_vocab", "rb"))
        embedding_dict = pickle.load(open(path + "/embeddings", "rb"))
        return embedding_dict, embedding_vocab


    def rereplace_puncuation_and_numbers(self, result, punctuation_list, numbers_list, removed_tokens):
        count_puncts = 0
        count_nums = 0
        for sentence in result:
            for word_label in sentence:
                if word_label[0] == "<PUNC/>":
                    word_label[0] = punctuation_list[count_puncts]
                    count_puncts = count_puncts + 1
                elif word_label[0] == "<NUM/>":
                    word_label[0] = numbers_list[count_nums]
                    count_nums = count_nums + 1
        for key, value in removed_tokens.items():
            i, j = key.split("_")
            result[int(i)].insert(int(j), [value, "REPLACED"])
        for i, sentence in enumerate(result):
            for j, word_label in enumerate(sentence):
                if word_label[1] == "REPLACED":
                    if j == 0 and j != len(sentence)-1 and sentence[j+1][1] not in self.begin_of_span():
                        word_label[1] = sentence[j+1][1]
                    elif j != len(sentence)-1 and sentence[j-1][1] == sentence[j+1][1] and sentence[j+1][1] not in self.begin_of_span():
                        word_label[1] = sentence[j+1][1]
                    elif j == len(sentence)-1 and sentence[j-1][1]:
                        word_label[1] = sentence[j-1][1]
                    else:
                        word_label[1] = "Token_Label.OUTSIDE"
        return result


    def predict(self, text):
        text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        input, sequence_lengths, punctuation_list, numbers_list, removed_tokens, real_length = self.prepare_sequence_labelling_text(texts=text)

        grouped_input = [input[n:n + 16] for n in range(0, len(input), 16)]
        grouped_sequence_lengths = [sequence_lengths[n:n + 16] for n in range(0, len(sequence_lengths), 16)]

        labels = []
        for i, group in enumerate(grouped_input):
            feed_dict = {
                self.input_tensor: group,
                self.dropout_keep_prob_tensor: 1.0,
                self.sequence_lengths_tensor: grouped_sequence_lengths[i]
            }

            labels.append(self.sess.run(self.arg_max, feed_dict))

        labels = [item for sublist in labels for item in sublist]
        index_to_word = {v: k for k, v in self.embedding_vocab.items()}
        labels = self.assign_labels(labels)
        if type == "argumentation":
            result = [[[index_to_word[word], labels[i][j]] for j, word in enumerate(sentence) if j < sequence_lengths[i]] for i, sentence in
                      enumerate(input) if i < real_length]
        else:
            result = [[[index_to_word[word], labels[i][0]] for j, word in enumerate(sentence) if j < sequence_lengths[i]] for i, sentence in
                      enumerate(input) if i < real_length]
        result = self.rereplace_puncuation_and_numbers(result, punctuation_list, numbers_list, removed_tokens)
        return result

def main():
    print("Started")


if __name__=='__main__':
    main()