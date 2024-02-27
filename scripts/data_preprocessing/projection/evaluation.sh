#!/usr/bin/env bash
lang=$1
corpus=../corpus

# perform the evaluation of a list of documents for each distance

for i in 45 11 # 11 12 15 16 41 42 45 46
do
	echo "i: "$i

	for filename in `ls $corpus/tags/en/original`
	do
		echo "f: "$filename
		filefolder="$corpus/tags/en/projected-to-$lang/"$i
		filepath="$corpus/tags/en/projected-to-$lang/"$i"/"$filename
		evaluation_file="../evaluation/"$filename".en-t$lang."$i".txt"
		python3 ./evaluate_projection.py "$corpus/tags/$lang/original/"$filename $filepath "$corpus/list_tags.txt" -aiwprf > $evaluation_file

		echo "finished: "$filename
	done

	filefolder="$corpus/tags/en/projected-to-$lang/"
	filepath="$corpus/tags/en/projected-to-$lang/"$i"/"
	evaluation_file="../evaluation/ALL.en-t$lang."$i".txt"
	python3 ./evaluate_projection.py "$corpus/tags/$lang/original/" $filepath "$corpus/list_tags.txt" -aiwprfd > $evaluation_file

	echo "finished: "$i
done
