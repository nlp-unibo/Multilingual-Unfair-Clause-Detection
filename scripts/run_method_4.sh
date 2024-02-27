corpus=$1

for lang in de it pl # en
do

for ngrams in 2 3
do

for tfidf in 0 1
do

resdir="experiments_bow_translated_joshua/$lang/linear_svm/"$ngrams"_grams_"$tfidf"_tfidf"
test -d $resdir || mkdirhier $resdir

for fold in 0 1 2 3 4
do

python3 scripts/linear_svm_transl.py $corpus/sentences/en/original $corpus/sentences/en/translated_joshua_from_$lang $corpus/tags/en/original $corpus/tags/$lang/original $corpus/list_tags.txt $corpus/lists/LIST_TRAIN_$fold.txt $corpus/lists/LIST_TEST_$fold.txt $ngrams $tfidf $resdir/y_pred_$fold.txt $resdir/y_true_$fold.txt > $resdir/log_$fold.txt

done

done

done

done

