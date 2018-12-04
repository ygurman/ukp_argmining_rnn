#! /bin/bash

echo "downloading glove embeddings from http://nlp.stanford.edu/data/glove.6B.zip"
wget http://nlp.stanford.edu/data/glove.6B.zip -P ../data
echo "extracting glove.6B.100d.txt"
mkdir ../data/glove.6B
unzip -j "../data/glove.6B.zip" "glove.6B.100d.txt" -d "../data/glove.6B"
