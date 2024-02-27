resdir=$1

for fold in 0 1 2 3 4
do

paste $resdir/y_true_$fold.txt $resdir/y_pred_$fold.txt | tr '.' ' ' | awk '{print $1,$3}'

done | computeResults.pl > $resdir/results_micro.txt

for fold in 0 1 2 3 4
do

paste $resdir/y_true_$fold.txt $resdir/y_pred_$fold.txt | tr '.' ' ' | awk '{print $1,$3}' | computeResults.pl

done | grep "(1)" | awk '{p+=$4;r+=$5;f+=$6}END{print p/NR,r/NR,f/NR}' > $resdir/results_macro.txt

