from spacy.lang.en import English
import sys

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    text = f.read()

nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp(text)
for s in doc.sents:
    print(s.text)


