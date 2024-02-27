# Example:
# lang=de
# bash segment.sh ../corpus/original_text/$lang ../corpus/sentences/$lang $lang

input_dir=$1
output_dir=$2
language=$3

for input_file in `ls $input_dir`
do

python3 segment_$language.py $input_dir/$input_file | grep -v "^ *$" | grep -v -P "^\t$" | grep -v -P "^ *\t*$" > $output_dir/$input_file

done

