
# Context-free grammars with NLTK
# Prof. Dr. Alexander Koller
# last accessed: 2023 Sep 13
# https://coli-saar.github.io/cl20/notebooks/CFGs.html

import nltk

groucho_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")

groucho_grammar.start()
groucho_grammar.productions()

sent = "I shot an elephant in my pajamas".split()
parser = nltk.ChartParser(groucho_grammar)
trees = list(parser.parse(sent))
print(nltk.TreePrettyPrinter(trees[0]).text())
print(nltk.TreePrettyPrinter(trees[1]).text())