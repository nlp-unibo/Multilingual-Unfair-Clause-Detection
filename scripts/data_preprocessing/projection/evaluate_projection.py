import sys
import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report


# from sklearn.metrics import multilabel_confusion_matrix


def estract_tag_list_from_file(tag_list, document_path):
    """
    Estracts the tags from a document. Return a list multi-label representation of the document.
    Each element in the list represents a line.
    Each line element is a list that contains a set of 0 and 1 values, one for each possible label.
    1 indicates the presence of a specific tag in that line. 0 indicates the absence of that tag.
    :param tag_list: list of tags that can be present in the document
    :param document_path: path of the document to be analyzed. Only tags are considered.
    :return: A list that contains the multi-label representation of the document
    """
    # for evaluation, the structure that is needed is the following:
    # array = [ [0, 1, 1, ...] , [0, 1, 0, ...] ]
    # each array element in the array represent a line
    # each element in the nested array represent the presence (1) or absence (0) of a given tag
    document_data = []
    with open(document_path) as f:
        for line in f:
            line_data = []
            labels = line.strip().lower().split()
            for tag in tag_list:
                if tag in labels:
                    line_data.append(1)
                else:
                    line_data.append(0)
            document_data.append(line_data)
    return document_data


parser = argparse.ArgumentParser(description="Evaluate a tag document or all the tag documents in a folder")
parser.add_argument("ground_truth_path", help="The path that contains the ground truth")
parser.add_argument("prediction_path", help="The path that will be evaluated")
parser.add_argument("tags_list_document", help="The path of the document with the list of tags")
parser.add_argument('-a', '--macro', help="Prints the f1-macro score", action="store_true")
parser.add_argument('-i', '--micro', help="Prints the f1-micro score", action="store_true")
parser.add_argument('-w', '--weighted', help="Prints the weighted f1-score", action="store_true")
parser.add_argument('-p', '--precision', help="Prints the precision (micro average)", action="store_true")
parser.add_argument('-r', '--recall', help="Prints the recall (micro average)", action="store_true")
parser.add_argument('-f', '--fullreport', help="Prints a full report with all the tags details", action="store_true")
parser.add_argument('-d', '--directory', help="Operates on an entire folder instead of a single document",
                    action="store_true")
# TODO: this should be implemented but there are problems with the library on the server
# parser.add_argument('-c', '--confusion', help="Prints the confusion matrices", action="store_true")

args = parser.parse_args()

ground_truth_tags_document = args.ground_truth_path
projected_tags_document = args.prediction_path
tag_list_document = args.tags_list_document

tag_list = []

# extract possible tag list
with open(tag_list_document) as f:
    for line in f:
        tag_list.append(line.strip().lower())
tag_list.sort()

real_tag_list = []

gt_documents = []
p_documents = []

if not args.directory:
    if os.path.isfile(args.ground_truth_path) and os.path.isfile(args.prediction_path):
        gt_documents = [args.ground_truth_path]
        p_documents = [args.prediction_path]
    else:
        print('ILLEGAL PATHS:\n\t' + args.ground_truth_path + '\n\t' + args.prediction_path)
        print('Aborting...')
        sys.exit()
else:
    # if a folder is selected
    if os.path.isdir(args.prediction_path) and os.path.isdir(args.ground_truth_path):
        p_documents_names = os.listdir(args.prediction_path)
        # for each file in the prediction folder
        for document_name in p_documents_names:
            # check if a corresponding ground truth file does exist
            gt_document = os.path.join(args.ground_truth_path, document_name)
            p_document = os.path.join(args.prediction_path, document_name)
            if os.path.isfile(gt_document) and os.path.isfile(p_document):
                gt_documents.append(gt_document)
                p_documents.append(p_document)
            else:
                print('ILLEGAL PATH:\n\t' + gt_document + '\n\t' + p_document)
                print('Skipping...')
    else:
        print('ILLEGAL PATHS:\n\t' + args.ground_truth_path + '\n\t' + args.prediction_path)
        print('Aborting...')
        sys.exit()

for gt_tag_document in gt_documents:
    # extract tags in the document
    with open(gt_tag_document) as f:
        tags = f.read().split()
        for tag in tags:
            if tag in tag_list and tag not in real_tag_list:
                real_tag_list.append(tag)

real_tag_list.sort()

prediction = []
ground_truth = []

for gt_document in gt_documents:
    document_ground_truth = estract_tag_list_from_file(real_tag_list, gt_document)
    ground_truth.extend(document_ground_truth)
for p_document in p_documents:
    document_prediction = estract_tag_list_from_file(real_tag_list, p_document)
    prediction.extend(document_prediction)

ground_truth = np.array(ground_truth)
prediction = np.array(prediction)

prfs = precision_recall_fscore_support(y_true=ground_truth, y_pred=prediction, average="micro")
precision = prfs[0]
recall = prfs[1]

if args.macro:
    print("F1-macro\t{0:.2f}".format(f1_score(y_true=ground_truth, y_pred=prediction, average="macro")))
if args.micro:
    print("F1-micro\t{0:.2f}".format(f1_score(y_true=ground_truth, y_pred=prediction, average="micro")))
if args.weighted:
    print("F1-weighted\t{0:.2f}".format(f1_score(y_true=ground_truth, y_pred=prediction, average="weighted")))
if args.precision:
    print("Precision\t{0:.2f}".format(precision))
if args.recall:
    print("Recall\t{0:.2f}".format(recall))
if args.fullreport:
    print(classification_report(y_true=ground_truth, y_pred=prediction, target_names=real_tag_list, digits=2))
# if args.confusion:
#    print(multilabel_confusion_matrix(y_true=ground_truth, y_pred=prediction, labels=tag_list))
