import textdistance
import sys

document_1 = sys.argv[1] # Automatically translated, tags known
document_2 = sys.argv[2] # Original, to be tagged
tags_document_1 = sys.argv[3] # Tags of document_1
which_distance = sys.argv[4]

#  DISTANCES 
#  1- hamming
#  2- levenshtein
#  3- damerau_levenshtein
#  4- needleman_wunsch
#  5- smith_waterman
#  6- jaccard
#  7- tanimoto
#  8- cosine
#  9- rle_ncd
# 10- entropy_ncd

if which_distance == '1':
    distance_computer = textdistance.hamming
elif which_distance == '2':
    distance_computer = textdistance.levenshtein
elif which_distance == '3':
    distance_computer = textdistance.damerau_levenshtein
elif which_distance == '4':
    distance_computer = textdistance.needleman_wunsch
elif which_distance == '5':
    distance_computer = textdistance.smith_waterman
elif which_distance == '6':
    distance_computer = textdistance.jaccard
elif which_distance == '7':
    distance_computer = textdistance.tanimoto
elif which_distance == '8':
    distance_computer = textdistance.cosine
elif which_distance == '9':
    distance_computer = textdistance.rle_ncd
elif which_distance == '10':
    distance_computer = textdistance.entropy_ncd
else:
    print('Unknown distance value: ' + str(which_distance))
    sys.exit()

sentences_document_tagged = []
sentences_document_to_be_tagged = []
original_sentences_document_to_be_tagged = []
labels_document_tagged = []

with open(document_1) as f:
    for line in f:
        sentences_document_tagged.append(line.strip().lower())

with open(document_2) as f:
    for line in f:
        sentences_document_to_be_tagged.append(line.strip().lower())
        original_sentences_document_to_be_tagged.append(line.strip())

with open(tags_document_1) as f:
    for line in f:
        labels_document_tagged.append(line.strip())

idx_1 = 0

for s_to_be_tagged in sentences_document_to_be_tagged:
    min_dist = 1e30
    idx_2 = 0
    for s_tagged in sentences_document_tagged:
        d = distance_computer.distance(s_to_be_tagged,s_tagged)
        if d < min_dist:
            min_dist = d
            idx_2_best = idx_2
        idx_2 = idx_2 + 1
    out = ''
    for tag in labels_document_tagged[idx_2_best].split(' '):
        if tag != '':
            out += tag + ' '
    print(out)
    idx_1 = idx_1 + 1



