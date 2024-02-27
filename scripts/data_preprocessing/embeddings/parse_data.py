import os

# for language in ["it", "de", "pl", "en"]:
for language in ["it", "de", "pl"]:
    data = []
    
    folderpath = os.path.join(os.getcwd(), "corpus2022", "sentences", language, "original")

    # folderpath = os.path.join(os.getcwd(), "corpus_NLLP2021", "sentences", language, "original")

    for (dirpath, dirnames, filenames) in os.walk(folderpath):
        for documentname in filenames:
            document = os.path.join(folderpath, documentname)
            with open(document) as f:
                doc1_data = f.readlines()
                doc1_data = list(map(lambda item: item.strip().lower(), doc1_data))
                data.extend(doc1_data)
    
    data.sort()
    
    # with open(os.path.join(os.getcwd(), "sentences_" + str(language) + ".txt"), "w") as f:
    with open(os.path.join(os.getcwd(), "sentences2022_" + str(language) + ".txt"), "w") as f:
        for sentence in data:
            f.write(sentence)
            f.write("\n")
    
    # with open(os.path.join(os.getcwd(), "CoNLL-U_" + str(language) + ".txt"), "w") as f:
    with open(os.path.join(os.getcwd(), "CoNLL-U2022_" + str(language) + ".txt"), "w") as f:
        for sentence in data:
            splitted = sentence.split()
            for num, token in enumerate(splitted, start=1):
                f.write(str(num))
                f.write("\t")
                f.write(token)
                f.write("\t")
                f.write(token)
                f.write("\n")
            f.write("\n")


for language in ["en"]:
    data = []
    
    for folder in ["original", "translated_from_de", "translated_from_it", "translated_from_pl", "translated_joshua_from_de", "translated_joshua_from_it", "translated_joshua_from_pl"]:
    
        folderpath = os.path.join(os.getcwd(), "corpus2022", "sentences", language, folder)

        for (dirpath, dirnames, filenames) in os.walk(folderpath):
            for documentname in filenames:
                document = os.path.join(folderpath, documentname)
                with open(document) as f:
                    doc1_data = f.readlines()
                    doc1_data = list(map(lambda item: item.strip().lower(), doc1_data))
                    data.extend(doc1_data)
        
        data.sort()
        
        with open(os.path.join(os.getcwd(), "sentences2022_" + str(language) + ".txt"), "w") as f:
            for sentence in data:
                f.write(sentence)
                f.write("\n")
        
        with open(os.path.join(os.getcwd(), "CoNLL-U2022_" + str(language) + ".txt"), "w") as f:
            for sentence in data:
                splitted = sentence.split()
                for num, token in enumerate(splitted, start=1):
                    f.write(str(num))
                    f.write("\t")
                    f.write(token)
                    f.write("\t")
                    f.write(token)
                    f.write("\n")
                f.write("\n")