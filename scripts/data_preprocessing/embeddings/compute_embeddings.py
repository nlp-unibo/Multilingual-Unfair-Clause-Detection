import numpy as np
import sys

sentences_file = sys.argv[1]
dictionary_file = sys.argv[2]

dictionary = np.load(dictionary_file, allow_pickle=True).item() 
dictionary = dict((key.strip(), value) for (key, value) in dictionary.items())

with open(sentences_file) as f:
    for line in f:
#        line = line.replace("\t", " ")
#        while "  " in line:
#            line = line.replace("  ", " ")
#        emb = dictionary[line.lower().strip()]
        line = line.lower()
        line = " ".join(line.split())
        #line = line + "\n"
        emb = dictionary[line]
        print([e for e in emb])

