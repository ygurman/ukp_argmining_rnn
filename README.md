# ukp_argmining_rnn

LSTM based e2e argument mining (AC segmentation, Classification and Relation Classification) trained on UKP's Annotated Essays V2 (https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp)

# running instructions:
1. setup -
 - packages used: pytorch, pydot, pandas, nltk (sentence tokenizer only)
 - clone the git
 - download stanford corenlp (to run as local server) and change first line in start_core_nlp_server.sh to match your downloawd location.
 - run the "download_glove_pretrained_embds.sh" [downloads embeddings
 - make sure the directory tree matches the git, as the preprocessing and data handeling depends on that

2. preprocess
 - run src/utils/preprocess.py with desired flags: #user@host python3 ./src/utils/preprocess.py [-c] [-b] [-v] [-d ../data]
  * -c : converts the whole dataset to CONLL-like tsv format (save to ./data/processed
  * -b : builds vocabularies (symbols to indices) and pretrained embeddings layer and stores them in ./data/vocabularies
  * -v : visualise all the dataset and saves it as PNG files in ./graphs
  * -d : data directory path

3. Segmentation and Classification
 - train:
  * run ./src/ac_tagging/train.py :  
    [-cp] configurations path {locatad at ./*.conf} for this model use either one of {"params.conf", "params_nopos.conf" (for no POS embeddings}
    [-m] operation mode (context scope - 's'entence 'p'aragraph or 'e'ssay)
    -> model (and every 25 epoch checkpoints) is saved to ./models
 - testing:
  * run ./src/ac_tagging/predict.py :
    [-cp] , [-m] {same as above}
    [-mp] path to the pretrained model 
    -> results saved to ./exps

4. Relations Classifications
 - train:
  * run ./src/relations/train.py 
    [-bl] - to train baseline model (explained in report - linear layer on top of dot multiplication of the SegmentorClassifier output for ACs). otherwise a f
   [-cp] as in the previous part - notice you use "params_rel*.conf" {"params_rel" - full model, params_rel_no_construct_features.conf" - not using structural embeddings. "params_rel_noenforce.conf" - not using previous traine segmentor-classifier to process input)
 - test:
  * run ./sec/relations/predict.py
   [-cp] [-mp] as above
   [-g] use gold segmentation (correct segmentation) instead of output #not implemented

** note: pretrained models are supplied in google drive link found in original report 
