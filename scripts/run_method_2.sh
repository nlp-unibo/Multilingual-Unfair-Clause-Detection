corpus=$1

ngrams=2
tfidf=0

for lang in de it pl
do

resdir="experiments_method_2/$lang"
test -d $resdir || mkdirhier $resdir

for fold in 0 1 2 3 4
do

python3 scripts/linear_svm_proj.py $corpus/sentences/$lang/original $corpus/tags/en/projected-to-$lang/46 $corpus/tags/$lang/original $corpus/list_tags.txt $corpus/lists/LIST_TRAIN_$fold.txt $corpus/lists/LIST_TEST_$fold.txt $ngrams $tfidf $resdir/y_pred_$fold.txt $resdir/y_true_$fold.txt > $resdir/log_$fold.txt

done

done

