ELMO

0) Download the language model in the XX_model folder (e.g., it_model) and change the config file (see the readme of ElmoForManyLang for the list of models and instructions for the config)
1) Use the "parse_data" script to format the corpus in the CoNLL format (change the paths as needed)
2) Run the models to compute embeddings, as in command example below (change the paths as needed)
3) Run the "converter" to change from hdf5 to npy (change the paths as needed)
4) Check everything with the "final_check" script

BERT
0) Perform the step for elmo (the sentences file is needed)
1) Run the "compute bert encoding" script
2) Check everything with the "final_check" script

Notes: during the preprocessing everything is lowercased and multiple spaces and tabs are replaced with single spaces






nohup python -m elmoformanylangs test \
    --input_format conll \
    --input /PATHTO/CoNLL-U2022_it.txt \
    --model /PATHTO/it_model/ \
    --output_prefix /PATHTO/elmo_embeddings2022_it \
    --output_format hdf5 \
    --output_layer -1 >log.it.txt 2>err.it.txt &