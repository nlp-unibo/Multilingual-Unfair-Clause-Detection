from deep_translator import GoogleTranslator
import sys

data_file = sys.argv[1]
lang = sys.argv[2]

with open(data_file) as f:
    for line in f:
        if (any(c.isalpha() for c in line)):
            print(GoogleTranslator(source='auto', target=lang).translate(line))
        else:
            print(line.strip())
