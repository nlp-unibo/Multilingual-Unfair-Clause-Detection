sentences_dir=$1
output_dir=$2
embeddings=$3

for x in `ls $sentences_dir`
do

echo $x
python3 compute_embeddings.py $sentences_dir/$x $embeddings > $output_dir/$x

done

