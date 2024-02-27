from spacy.lang.pl import Polish
import sys

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    text = f.read()

nlp = Polish()
nlp.add_pipe("sentencizer")
doc = nlp(text)
for s in doc.sents:
    print(s.text)


