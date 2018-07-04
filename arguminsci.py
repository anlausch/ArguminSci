import warnings
warnings.filterwarnings("ignore")

import argparse
import codecs
import serve

parser = argparse.ArgumentParser(description='Analyze Argumentation and Discourse Structure in Scientific Writing.')
parser.add_argument('inputfile', type=str, help='The name of the textual file containing the input text.')
parser.add_argument('outputfile', type=str, help='The name of the textual file where the output should be stored.')

args = parser.parse_args()
print("ArguminSci started")
print("Reading " + args.inputfile)
text = ""
with codecs.open(args.inputfile, "r", "utf8") as input:
    text = input.read()
    input.close()
print("Input loaded")
print("Load model")
model = serve.load_model()
embd_dict, embedding_vocab = serve.load_embeddings()
print("Model loaded")
print("Predict")
result = serve.predict(text=text, embd_vocab=embedding_vocab, model=model)
print("Predicted")
print("Writing output")
with codecs.open(args.outputfile, "w", "utf8") as output:
    output.write(str(result))
    output.close()
print("Saved output in " + args.outputfile)