import textdistance
import sys
import argparse
import scipy.spatial.distance as distances
import numpy as np
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from fastdtw import fastdtw, dtw

parser = argparse.ArgumentParser(description="Project tags from a source textual document, "
                                             "in which the labels are specified, to another textual document, "
                                             "for which the tags are unknown.")
parser.add_argument("source_text", help="The path of the sentences document for which the tags are known")
parser.add_argument("target_text", help="The path of the sentences document for which the tags are unknown")
parser.add_argument("source_tags", help="The path of the tags document related to the first sentence document")

parser.add_argument('-e', '--embeddings',
                    help="The path of a file which contains pre-loaded embeddings")

parser.add_argument('-d', '--distance', type=int,
                    choices=[11, 12, 13, 14, 15, 16,
                             21, 22, 23, 24, 25, 26,
                             31, 32, 33, 34, 35, 36,
                             41, 42, 43, 44, 45, 46,
                             111, 112,
                             121, 122,
                             131, 132,
                             141, 142],
                    help="distance used for matching", default="11")

parser.add_argument('-u', '--update', help="Updates the embeddings file if new ones are found", action="store_true")

parser.add_argument('-n', '--noelmo',
                    help="For use with only pre-trained word embeddings and no tensorflow/elmo libraries",
                    action="store_true")

args = parser.parse_args()

document_1 = args.source_text  # Automatically translated, tags known
document_2 = args.target_text  # Original, to be tagged
tags_document_1 = args.source_tags  # Tags of document_1
embeddings_file = args.embeddings
update = args.update
noelmo = args.noelmo
which_distance = int(args.distance)

if args.embeddings is not None and os.path.isfile(embeddings_file):
    dictionary = np.load(embeddings_file,allow_pickle=True).item()
else:
    dictionary = {}

#  DISTANCES
#  11- cosine
#  12- cosine plus normalization
#  13- cosine plus pca
#  14- cosine plus normalization and pca
#  21- canberra
#  22- canberra plus normalization
#  23- canberra plus pca
#  24- canberra plus normalization and pca

# <100 - use of FDTW
# >100 - use of real DTW


cosine_distances = [11, 12, 13, 14, 15, 16, 111, 112]
canberra_distances = [21, 22, 23, 24, 25, 26, 121, 122]
euclidean_distances = [31, 32, 33, 34, 35, 36, 131, 132]
braycurtis_distances = [41, 42, 43, 44, 45, 46, 141, 142]
fdtw_distances = [15, 16, 25, 26, 35, 36, 45, 46]
ndtw_distances = [111, 112, 121, 122, 131, 132, 141, 142]

normalization_distances = [12, 14, 16, 22, 24, 26, 32, 34, 36, 42, 44, 46]
pca_distances = [13, 14, 23, 24, 33, 34, 43, 44]

if which_distance in cosine_distances:
    distance_computer = distances.cosine
elif which_distance in canberra_distances:
    distance_computer = distances.canberra
elif which_distance in euclidean_distances:
    distance_computer = distances.euclidean
elif which_distance in braycurtis_distances:
    distance_computer = distances.braycurtis
else:
    print('Unknown distance value: ' + str(which_distance))
    sys.exit()

sentences_document_tagged = []
sentences_document_to_be_tagged = []
labels_document_tagged = []


# create ELMO model
# my_model = CustomModel()
# elmo = hub.eval_function_for_module("https://tfhub.dev/google/elmo/2")


def parse_sentences(data, noelmo):
    parsed_data = []
    global dictionary

    if noelmo:
        for sentence in tqdm(data):
            try:
                sentence = (' '.join(sentence.split())).strip()
                if sentence in dictionary.keys():
                    embedding = dictionary[sentence]
                else:
                    sentence = sentence + '\n'
                    embedding = dictionary[sentence]
                parsed_data.append(embedding)
            except:
                print("Sentence non found:")
                print(sentence)
                sys.exit()
    else:
        import tensorflow_hub as hub
        with hub.eval_function_for_module("https://tfhub.dev/google/elmo/2") as elmo:
            for sentence in tqdm(data):
                if sentence in dictionary.keys():
                    embedding = dictionary[sentence]
                else:
                    batch_in = np.array([sentence])
                    embedding = elmo(batch_in, as_dict=True)['default'][0]
                    embedding = np.ravel(embedding)
                    if update:
                        dictionary[sentence] = embedding
                parsed_data.append(embedding)

    parsed_data = np.array(parsed_data)
    return parsed_data


with open(document_1) as f:
    doc1_data = f.readlines()
    doc1_data = list(map(lambda item: item.strip().lower(), doc1_data))

sentences_document_tagged = parse_sentences(doc1_data, noelmo)

with open(document_2) as f:
    doc2_data = f.readlines()
    doc2_data = list(map(lambda item: item.strip().lower(), doc2_data))

sentences_document_to_be_tagged = parse_sentences(doc2_data, noelmo)

if update and embeddings_file is not None:
    np.save(embeddings_file, dictionary)

with open(tags_document_1) as f:
    for line in f:
        labels_document_tagged.append(line.strip())

# PCA STEP
if which_distance in pca_distances:
    pca = PCA(n_components=50)
    sentences_document_tagged = pca.fit_transform(sentences_document_tagged)
    pca = PCA(n_components=50)
    sentences_document_to_be_tagged = pca.fit_transform(sentences_document_to_be_tagged)

# NORMALIZATION STEP
if which_distance in normalization_distances:
    sentences_document_tagged = normalize(sentences_document_tagged, axis=0)
    sentences_document_to_be_tagged = normalize(sentences_document_to_be_tagged, axis=0)

matches = dict()
total_distance = 0
# USE OF DTW ALGORITHM
if which_distance in fdtw_distances or which_distance in ndtw_distances:

    if which_distance in fdtw_distances:
        dtw_distance = fastdtw
    else:
        dtw_distance = dtw

    total_distance, path = dtw_distance(sentences_document_to_be_tagged, sentences_document_tagged,
                                        dist=distance_computer)
    # print(path) # DEBUG

    prev_idx_1 = 0
    out = ""
    for couple in tqdm(path):
        idx_1 = couple[0]
        idx_2 = couple[1]
        if prev_idx_1 != idx_1:  # if the sentence to be tagged has changed, print the labels of the previous one
            print(out)
            out = ""
        prev_idx_1 = idx_1
        couple_distance = distance_computer(sentences_document_to_be_tagged[idx_1],
                                            sentences_document_tagged[idx_2])
        matches[(couple)]=couple_distance

        # tag = doc1_data[idx_2] # DEBUG
        # out += tag + " --- " # DEBUG
        for tag in labels_document_tagged[idx_2].split(' '):
            if tag != '':
                out += tag + ' '
    print(out)
else:  # USE OF NORMAL DISTANCE
    try:
        # 1 to N procedure: each sentence to be tagged is matched with 1 tagged sentence
        idx_1 = 0
        for s_to_be_tagged in tqdm(sentences_document_to_be_tagged):
            min_dist = 1e30
            idx_2 = 0
            for s_tagged in sentences_document_tagged:
                d = distance_computer(s_to_be_tagged, s_tagged)
                if d < min_dist:
                    min_dist = d
                    idx_2_best = idx_2
                idx_2 = idx_2 + 1
            out = ''
            for tag in labels_document_tagged[idx_2_best].split(' '):
                if tag != '':
                    out += tag + ' '
            print(out)
            matches[(idx_1, idx_2_best)]=min_dist
            idx_1 = idx_1 + 1
    except Exception as e:
        print(len(sentences_document_tagged))  # DEBUG
        print(len(labels_document_tagged))  # DEBUG
        print(len(sentences_document_to_be_tagged))  # DEBUG
        print('Error: ' + str(e))

for key in matches.keys():
    print(str(key) + "\t" + str(matches[key]), file=sys.stderr)
