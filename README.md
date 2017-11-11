# Abstractive-Summarization-using-Query-based-Deep-Neural-Attention-Models
System for generating meaningful summaries of retrieved documents, specific to query text using Deep Attention Models

## Requirements:
* [tensorflow-0.12](https://www.tensorflow.org/versions/r0.12/get_started/os_setup)
* [gensim](https://pypi.python.org/pypi/gensim)
* [nltk](http://www.nltk.org/install.html)

## Preprocessing
    * cd src/dataextraction_scripts
    * python make_folds.py ../../data <number_of_folds> <new_dir_for_10_folds> 
    * By default run : python make_folds.py ../../data 10 ../../data

## Get the Glove embeddings:
    mkdir Embedding
    cd Embedding
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    echo 2196017 300 > temp_metadata
    cat temp_metadata glove.840B.300d.txt > embeddings
    rm temp_metadata
 ## Train the model
      sh ./train.sh config.txt
 ## Test Inference:
     sh ./test.sh config.txt output/test_final_results


