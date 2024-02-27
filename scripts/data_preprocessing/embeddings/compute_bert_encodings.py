from transformers import *
import torch
import numpy as np
import os
from tqdm import tqdm


models = {"it": "dbmdz/bert-base-italian-xxl-cased",
          "en": "bert-base-uncased",
		  "de": "bert-base-german-dbmdz-uncased",
		  "pl": "dkleczek/bert-base-polish-uncased-v1",
		  }


for language in ["it", "de", "pl", "en"]:
    model_name = models[language]
    embeddings_file = "bert_embeddings_many2022_" + language + ".npy"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
	
    
    # with open(os.path.join(os.getcwd(), "prova_sentences.txt"), "r") as f:
    with open(os.path.join(os.getcwd(), "sentences2022_" + str(language) + ".txt"), "r") as f:
        sentences = f.readlines()
    
    try:
        newdict = np.load(embeddings_file, allow_pickle=True).item()    
        print("Dictionary loaded")
        print(len(newdict.keys()))
    except:
        print("Failed to load dictionary")
        newdict = {}
	
    for count, sentence in enumerate(tqdm(sentences, total=len(sentences), mininterval=5)):
    
        sentence = sentence.lower()
        sentence = " ".join(sentence.split())
    
        if sentence not in newdict.keys():
            encoded_input = tokenizer(sentence, return_tensors='pt')
            
            with torch.no_grad():
                output = model(**encoded_input)
            
            np_output = np.array(output["pooler_output"][0])
            
            newdict[sentence] = np_output
            if count % 100 == 0:
                np.save(embeddings_file, newdict)
    
    np.save(embeddings_file, newdict)
    dictionary = np.load(embeddings_file, allow_pickle=True).item()
    print("Final size:")
    print(len(dictionary.keys()))
    print("Size of one element:")
    print(np.shape(dictionary[list(dictionary.keys())[0]]))
    
    with open(os.path.join(os.getcwd(), "BERTdictionary2022_" + str(language) + ".txt"), "w") as f:
        keys = list(dictionary.keys())
        keys.sort()
        for key in keys:
            f.write(key)
            f.write("\n")

