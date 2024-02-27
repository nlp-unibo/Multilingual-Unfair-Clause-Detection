import h5py
import numpy as np
import os
from tqdm import tqdm


for language in ["it", "pl", "de", "en"]:

    sentences = []

    with open(os.path.join(os.getcwd(), "sentences2022_" + str(language) + ".txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\t", " ")
            while "  " in line:
                line = line.replace("  ", " ")
            sentences.append(line)

    h = h5py.File('elmo_embeddings2022_' + language + '.ly-1.hdf5')
    
    embeddings_file = "elmo_embeddings_many2022_" + language + ".npy"
    
    try:
        newdict = np.load(embeddings_file, allow_pickle=True).item()    
        print("Dictionary loaded")
        print(len(newdict.keys()))
    except:
        print("Failed to load dictionary")
        newdict = {}
    
    keylist = h.keys()

    for count, key in enumerate(tqdm(keylist, total=len(keylist), mininterval=5)):

        newkey = key.replace("\t", " ")
        newkey = newkey.replace("$period$", ".")
        newkey = newkey.replace("$backslash$", "/")
        
        
        if newkey+"\n" not in sentences:
            print("ERROR: missing sentence!")
            print("\t" + newkey)
            print("\t\t" + str(newkey.encode()))
            
            
        if newkey not in newdict.keys():
            nparray = np.array(list(h[key]))
            
            newvalue = np.average(nparray, axis=0)
            
            newdict[newkey] = newvalue
            if count % 100 == 0:
                np.save(embeddings_file, newdict)
        


    np.save(embeddings_file, newdict)
    
    dictionary = np.load(embeddings_file, allow_pickle=True).item()
    print("Final size:")
    print(len(dictionary.keys()))
    print("Size of one element:")
    print(np.shape(dictionary[list(dictionary.keys())[0]]))
    
    with open(os.path.join(os.getcwd(), "dictionary2022_" + str(language) + ".txt"), "w") as f:
        keys = list(dictionary.keys())
        keys.sort()
        for key in keys:
            f.write(key)
            f.write("\n")

