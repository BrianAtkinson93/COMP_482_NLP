from stanfordcorenlp import StanfordCoreNLP as nlp
from nltk.tree import Tree
import json

# needed to install stanfordcorenlp

# https://stanfordnlp.github.io/CoreNLP/download.html
# Navigate to the Stanford CoreNLP install folder and run the CoreNLPServer
#   so, e.g.: folder C:\Folders\Projects\Python\Stanford\stanford-corenlp-4.5.4
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# CTRL-C shuts the server down fine

# https://nlp.stanford.edu/software/parser-faq.html
# see the FAQ for memory issues involving -mx4g option

# if you just want to parse some English text
#   java edu.stanford.nlp.process.PTBTokenizer inputFile > outputFile
# https://nlp.stanford.edu/software/parser-faq.html#t

parser = nlp('http://localhost', port=9000)

text = 'Victoria is the capital of British Colubmia; the Empress Hotel is in its downtown waterfront area.'
ann = parser.annotate(
	text,
	properties={
		'annotators': 'tokenize,ssplit,pos,depparse,parse',
		'outputFormat': 'json'
	}
)
parsed = json.loads(ann)
parse_tree = [s['parse'] for s in parsed['sentences']]
print(parse_tree)
Tree.fromstring(parse_tree[0]).pretty_print()
