import MyPOSTagger

#######################################################################
## Build structure and required variables of POSTagger for Training :##
train = MyPOSTagger.Train('POST-Persian-Corpus-Train.txt', 'POST-Persian-Corpus-Test.txt')
train.Read_data()

## Compute Requiered probs of viterbi algorithm and save them in text files ##
train.CalculateEmisProb()
train.CalculateTransProb()
train.CalculateInitProb()