#!/bin/bash
corpus=../corpus
lang=$1

# perform the evaluation of a list of documents for each distance

for i in 46 #11 12 15 16 41 42 45 46
do
	for filename in `ls $corpus/sentences/en/original`
	do
		echo "i: "$i
		filefolder="$corpus/tags/en/projected-to-$lang/"$i
		filepath="$corpus/tags/en/projected-to-$lang/"$i"/"$filename
		mkdirhier $filefolder
		echo $filepath
		python3.7 ./elmo_project_tags.py "$corpus/sentences/en/original/"$filename "$corpus/sentences/en/translated_from_$lang/"$filename "$corpus/tags/en/original/"$filename -d $i -e "$corpus/embeddings_npy/bert_embeddings_many2022_en.npy" -n > $filepath 2> "../logs/err.en-t$lang."$i"."$filename
		echo "finished"
	done
done
