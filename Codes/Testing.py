import MyPOSTagger

#######################################################################
## Build structure and required variables of POSTagger for Testing :##

test = MyPOSTagger.Test('POST-Persian-Corpus-Test.txt')
test.Read_data()
test.tag_allsentences() ## Here we used MyViterbi function for each sentences ...