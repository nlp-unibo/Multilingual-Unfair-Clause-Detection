import os
import h5py
import numpy as np
import os
from tqdm import tqdm



for language in ["it", "de", "pl", "en"]:

    for embedding in ["elmo", "bert"]:
        
        embeddings_file = embedding + "_embeddings_many2022_" + language + ".npy"

        dictionary = np.load(embeddings_file, allow_pickle=True).item()
        dict_keys = dictionary.keys()
        
        
        if language == "en":
            folders = ["original", "translated_from_de", "translated_from_it", "translated_from_pl", "translated_joshua_from_de", "translated_joshua_from_it", "translated_joshua_from_pl"]
        else:
            folders = ["original"]
        
        for folder in folders:
            
            folderpath = os.path.join(os.getcwd(), "corpus2022", "sentences", language, folder)

            for (dirpath, dirnames, filenames) in os.walk(folderpath):
                for documentname in filenames:
                    document = os.path.join(folderpath, documentname)
                    with open(document) as f:
                        doc1_data = f.readlines()
                        doc1_data = list(map(lambda item: item.strip().lower(), doc1_data))
                    
                        for line in doc1_data:
                            line = " ".join(line.split())
                            if  line not in dict_keys and line +"\n" not in dict_keys and len(line)>1:
                                print("ERROR!\t" + str(embedding) + "\t" + str(document) + "\n\t" + str(line))