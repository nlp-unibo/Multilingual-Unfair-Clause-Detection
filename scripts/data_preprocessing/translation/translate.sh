target=pl

for source in en #de it pl
do

for x in `ls ../corpus_2022/sentences/$source/original`
do

python3.7 translate.py ../corpus_2022/sentences/$source/original/$x $target > ../corpus_2022/sentences/$target/translated_from_$source/$x

done

done

