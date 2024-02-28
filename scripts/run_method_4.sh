corpus=$1

ngrams=2
tfidf=0

for lang in de it pl
do

for translation in google opus joshua
do

resdir="experiments_method_4_$translation/$lang/"
test -d $resdir || mkdirhier $resdir

for fold in 0 1 2 3 4
do

python3 scripts/linear_svm_transl.py $corpus/sentences/en/original $corpus/sentences/en/translated_from_$lang"_"$translation $corpus/tags/en/original $corpus/tags/$lang/original $corpus/list_tags.txt $corpus/lists/LIST_TRAIN_$fold.txt $corpus/lists/LIST_TEST_$fold.txt $ngrams $tfidf $resdir/y_pred_$fold.txt $resdir/y_true_$fold.txt > $resdir/log_$fold.txt

done

done

done

