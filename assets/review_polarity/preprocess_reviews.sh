#!/bin/bash

# unzip and unpack the tar file
# gunzip -c movie_data.tar.gz | tar xopf -

cd txt_sentoken

mkdir preprocess_data

# puts four files in the combined_files directory:
# full_train.txt, full_test.txt, original_train_ratings.txt, and original_test_ratings.txt
for split in train test;
do
  for sentiment in pos neg;
  do 
    for file in $split/$sentiment/*; 
    do
      cat $file >> movie_data/full_${split}.txt; 
      echo >> movie_data/full_${split}.txt; 
	    # This line adds files containing the original reviews if desired
             # echo $file | cut -d '_' -f 2 | cut -d "." -f 1 >> combined_files/original_${split}_ratings.txt; 
    done;
  done;
done;
