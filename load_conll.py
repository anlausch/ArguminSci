import os
import codecs
import numpy as np

#class CoNLL_Sentence:
#    def __init__(self, tokens):
#        self.tokens = tokens

#class Simplified_CoNLL_Token:
#    def __init__(self, token, token_label, sentence_label):
#        self.token = token
#        self.token_label = token_label
#        self.sentence_label = sentence_label

def parse_conll_file(file, multiple=False):
    tokens = []
    sentences = []
    for line in file.readlines():
        if(line == "\n"):
            if len(tokens) != 0:
                #sentence = CoNLL_Sentence(tokens=tokens)
                sentences.append(tokens)
                tokens = []
            else:
                print("That should not happen.")
        else:
            parts = line.split("\t")
            if multiple == False:
                token = [parts[0], parts[1], parts[2]]#Simplified_CoNLL_Token(token=parts[0], token_label=parts[1], sentence_label=parts[2])
            else:
                token = [parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]]
            tokens.append(token)
    return sentences


def parse_conll_files(path, multiple=False):
    sentences = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with codecs.open(os.path.join(subdir, file), "r", "utf8") as f:
                file_sentences = parse_conll_file(f, multiple=multiple)
                sentences.append(file_sentences)
    return sentences


def transform_to_model_input(sentences):
    x = []
    y_arg = []
    y_rhet = []
    for sentence in sentences:
        x_sentence = []
        y_sentence_arg = []
        y_sentence_rhet = []
        for token in sentence:
            x_sentence.append(token[0])
            y_sentence_arg.append(token[1])
            y_sentence_rhet.append(token[2])
        x.append(np.array(x_sentence))
        y_arg.append(np.array(y_sentence_arg))
        y_rhet.append(np.array(y_sentence_rhet))
    return np.array(x), np.array(y_arg), np.array(y_rhet)


def transform_to_model_input_multiple(sentences):
    x = []
    y_arg = []
    y_rhet = []
    y_aspect = []
    y_summary = []
    y_citation = []
    for sentence in sentences:
        x_sentence = []
        y_sentence_arg = []
        y_sentence_rhet = []
        y_sentence_aspect = []
        y_sentence_summary = []
        y_sentence_citation = []
        for token in sentence:
            x_sentence.append(token[0])
            y_sentence_arg.append(token[1])
            y_sentence_rhet.append(token[2])
            y_sentence_aspect.append(token[3])
            y_sentence_summary.append(token[4])
            y_sentence_citation.append(token[5])
        x.append(np.array(x_sentence))
        y_arg.append(np.array(y_sentence_arg))
        y_rhet.append(np.array(y_sentence_rhet))
        y_aspect.append(np.array(y_sentence_aspect))
        y_summary.append(np.array(y_sentence_summary))
        y_citation.append(np.array(y_sentence_citation))
    return np.array(x), np.array(y_arg), np.array(y_rhet), np.array(y_aspect), np.array(y_summary), np.array(y_citation)


def load_data(path="./../annotations_conll"):
    sentences = parse_conll_files(path)
    flat_sentences = [item for sublist in sentences for item in sublist]
    x, y_arg, y_rhet = transform_to_model_input(flat_sentences)
    print("Data size: " + str(len(x)))
    return x, y_arg, y_rhet


def load_data_multiple(path=""):
    sentences = parse_conll_files(path, multiple=True)
    flat_sentences = [item for sublist in sentences for item in sublist]
    x, y_arg, y_rhet, y_aspect, y_summary, y_citation = transform_to_model_input_multiple(flat_sentences)
    print("Data size: " + str(len(x)))
    return x, y_arg, y_rhet, y_aspect, y_summary, y_citation


def main():
    print("Process started")
    sentences = parse_conll_files("./annotations_conll")
    flat_sentences = [item for sublist in sentences for item in sublist]
    x, y_arg, y_rhet = transform_to_model_input(flat_sentences)
    print("Process ended")

if __name__ == "__main__":
    main()