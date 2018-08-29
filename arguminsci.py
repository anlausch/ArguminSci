import warnings
warnings.filterwarnings("ignore")

import argparse
import codecs
import serve
import os

parser = argparse.ArgumentParser(description='Analyze Argumentation and Rhetorical Aspects in Scientific Writing.')
parser.add_argument('inputfile', type=str, help='The name of the textual file containing the input text.')
parser.add_argument('outputfolder', type=str, help='The name of the output folder where the output should be stored.')

parser.add_argument("--argumentation", help="Extract argument components.", action="store_true")
parser.add_argument("--discourse", help="Analyze discourse roles.", action="store_true")
parser.add_argument("--aspect", help="Analyze subjective aspects.", action="store_true")
parser.add_argument("--citation", help="Extract citation contexts", action="store_true")
parser.add_argument("--summary", help="Assign summary relevance.", action="store_true")

args = parser.parse_args()
if not args.argumentation and not args.discourse and not args.aspect and not args.citation and not args.summary:
    parser.print_help()
    exit()

print("ArguminSci started")
print("Reading " + args.inputfile)
text = ""
with codecs.open(args.inputfile, "r", "utf8") as input:
    text = input.read()
    input.close()
print("Input loaded")
print("Load model(s)")
if args.argumentation:
    argumentation_model = serve.Model("argumentation")
if args.discourse:
    discourse_model = serve.Model("discourse")
if args.aspect:
    aspect_model = serve.Model("aspect")
if args.citation:
    citation_model = serve.Model("citation")
if args.summary:
    summary_model = serve.Model("summary")
print("Model(s) loaded")
print("Predict")
if args.argumentation:
    argumentation = argumentation_model.predict(text)
if args.discourse:
    discourse = discourse_model.predict(text)
if args.aspect:
    aspect = aspect_model.predict(text)
if args.citation:
    citation = citation_model.predict(text)
if args.summary:
    summary = summary_model.predict(text)
print("Predicted")
print("Writing output")
if not os.path.exists(args.outputfolder):
    os.makedirs(args.outputfolder)

if args.argumentation:
    with codecs.open(os.path.join(args.outputfolder, "argumentation.txt"), "w", "utf8") as output:
        output.write(str(argumentation))
        output.close()
if args.discourse:
    with codecs.open(os.path.join(args.outputfolder, "discourse.txt"), "w", "utf8") as output:
        output.write(str(discourse))
        output.close()
if args.aspect:
    with codecs.open(os.path.join(args.outputfolder, "aspect.txt"), "w", "utf8") as output:
        output.write(str(aspect))
        output.close()
if args.citation:
    with codecs.open(os.path.join(args.outputfolder, "citation.txt"), "w", "utf8") as output:
        output.write(str(citation))
        output.close()
if args.summary:
    with codecs.open(os.paths.join(args.outputfolder, "summary.txt"), "w", "utf8") as output:
        output.write(str(summary))
        output.close()
print("Saved output in " + args.outputfolder)