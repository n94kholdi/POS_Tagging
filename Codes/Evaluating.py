import MyPOSTagger

def CalculatePOSTAccuracy(file_test, file_predict):

    ## Read test file and extract tags
    test = MyPOSTagger.Test(file_test)
    test.Read_data()
    test_tags = test.All_tags

    ## Read My predict file and extract tags
    Mypred = MyPOSTagger.Test(file_predict)
    Mypred.Read_data()
    Mypred_tags = Mypred.All_tags

    correctly_tagged = 0
    all_words = len(Mypred_tags)
    for t in range(0, len(Mypred_tags)):
        correctly_tagged += (Mypred_tags[t] == test_tags[t])

    print('Number of correctly_tagged : ', correctly_tagged)
    print('Number of all words :', all_words)
    print('Accuracy :', correctly_tagged / all_words)



CalculatePOSTAccuracy('POST-Persian-Corpus-Test.txt', 'POST-Persian-Corpus-Test-MyOut.txt')


