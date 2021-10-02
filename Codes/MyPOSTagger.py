import ast
import codecs
import numpy as np
from collections import Counter
wordDict = Counter()


class Train():

    def __init__(self, train_file, test_file):
        super().__init__()

        self.train_file = train_file
        self.test_file = test_file
        self.Tags = {}
        self.All_Words = {}
        self.lines = []

        self.word_and_tag = {}
        self.tag_and_tag = {}


    def Read_data(self):

        ### Read Train data :
        with codecs.open('Data//' + self.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split()

                if len(text) > 2:
                    clean_text = text[0]
                    for i in range(1, len(text)-1):
                        clean_text += '\u200c' + text[i]
                    text = [clean_text, text[-1]]

                self.lines.append(text)
                if text != ['.', 'DELM']:
                    ## update words :
                    if text[0] in self.All_Words.keys():
                        self.All_Words[text[0]] += 1
                    else:
                        self.word_and_tag[text[0]] = {}
                        self.All_Words[text[0]] = 0
                        self.All_Words[text[0]] += 1
                    if len(text) == 1:
                        a = 0
                    ## update Tags :
                    if text[1] in self.Tags.keys():
                        self.Tags[text[1]] += 1
                    else:
                        self.Tags[text[1]] = 0
                        self.Tags[text[1]] += 1
                else:
                    a = 0

        return self.lines

    def Build_Unique_words(self):

        test = Test(self.test_file)
        test.Read_data()

        for word in test.Words_test.keys():
            if word not in self.All_Words.keys():
                self.All_Words[word] = 0

    ### compute probability for viterbi algorithm

    def CalculateEmisProb(self):

        ##########################################################
        ## Build word_and_tag dictionary by all words and tags ###

        for word in self.All_Words.keys():
            self.word_and_tag[word] = {}
            for tag in self.Tags.keys():
                self.word_and_tag[word][tag] = 0

        ###################################################
        ## Updata word_and_tag entries by train dataset ###

        self.Build_Unique_words()

        for i in range(len(self.lines)):

            line1 = self.lines[i] ## contain word and tag

            if len(line1) > 1 and line1 != ['.', 'DELM']:
                self.word_and_tag[line1[0]][line1[1]] += (1 / (self.Tags[line1[1]]))


        with open('Probs/ProbsB.txt', 'w') as file:
            file.write(str(self.word_and_tag))

    def CalculateTransProb(self):

        self.tag_and_tag['start'] = {}
        self.tag_and_tag['start'][self.lines[0][1]] = 0
        self.tag_and_tag['start'][self.lines[0][1]] += 1

        for i in range(len(self.lines)-1):

            line1 = self.lines[i] ## contain word and tag
            line2 = self.lines[i+1]

            if (len(line2) > 1) & (len(line1) > 1):

                if line1[1] in self.tag_and_tag.keys():
                    if line2[1] in self.tag_and_tag[line1[1]]:
                        self.tag_and_tag[line1[1]][line2[1]] += (1/self.Tags[line1[1]])
                    else:
                        self.tag_and_tag[line1[1]][line2[1]] = 0
                        self.tag_and_tag[line1[1]][line2[1]] += (1/self.Tags[line1[1]])
                else:
                    self.tag_and_tag[line1[1]] = {}

            if line1[0] == '.':
                if len(line2) > 1:
                    if line2[1] in self.tag_and_tag['start']:
                        self.tag_and_tag['start'][line2[1]] += 1
                    else:
                        self.tag_and_tag['start'][line2[1]] = 0
                        self.tag_and_tag['start'][line2[1]] += 1

            if line2[0] == '.':
                if len(line1) > 1:
                    if 'end' in self.tag_and_tag[line1[1]]:
                        self.tag_and_tag[line1[1]]['end'] += 1
                    else:
                        self.tag_and_tag[line1[1]]['end'] = 0
                        self.tag_and_tag[line1[1]]['end'] += 1

        with open('Probs/ProbsA.txt', 'w') as file:
            file.write(str(self.tag_and_tag))

    def CalculateInitProb(self):

        with open('Probs/ProbsPi.txt', 'w') as file:
            file.write(str(self.Tags))

########################################################################
########################################################################

class Test():
    def __init__(self, filename):
        super().__init__()

        self.filename = filename
        self.Tags_test = {}
        self.Words_test = {}
        self.lines_test = []

        self.Tags = self.load_probs('ProbsPi.txt')
        self.tag_and_tag = self.load_probs('ProbsA.txt')
        self.word_and_tag = self.load_probs('ProbsB.txt')
        self.All_sentences = []

        self.All_tags = []
        self.tag_predicted = []

    def Read_data(self):

        ### Read Test data :
        with codecs.open('Data//' + self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split()
                # if text[0] != '##########':

                if len(text) > 2:
                    clean_text = text[0]
                    for i in range(1, len(text) - 1):
                        clean_text += '\u200c' + text[i]
                    text = [clean_text, text[-1]]

                self.lines_test.append(text)
                if text != ['.', 'DELM']:

                    self.All_tags.append(text[1])
                    ## update words :
                    if text[0] in self.Words_test.keys():
                        self.Words_test[text[0]] += 1
                    else:
                        # self.word_and_tag[text[0]] = {}
                        self.Words_test[text[0]] = 0
                        self.Words_test[text[0]] += 1
                    if len(text) == 1:
                        a = 0
                    ## update Tags :
                    if text[1] in self.Tags_test.keys():
                        self.Tags_test[text[1]] += 1
                    else:
                        self.Tags_test[text[1]] = 0
                        self.Tags_test[text[1]] += 1
                else:
                    a = 0
        with open('Data/POST-Persian-Corpus-Test-Sent.txt', 'w') as f:
            for line in self.lines_test:
                f.write(line[0] + '\n')
        f.close()

    def load_probs(self, filename):

        file = open("Probs//" + filename, "r")
        contents = file.read()
        probs = ast.literal_eval(contents)
        file.close()

        return probs

    def MyViterbi(self, sentence):

        All_Tags = list(self.Tags.keys())
        best_tag_sequence = np.zeros((len(All_Tags), len(sentence)))

        ### dynamic matrix for find best sequence
        dynamic_mat = np.zeros((len(All_Tags), len(sentence)))

        ### compute Initialized values
        for t in range(0, len(All_Tags)):
            if sentence[0] in self.word_and_tag.keys():
                if (All_Tags[t] in self.tag_and_tag['start'].keys()) & (All_Tags[t] in self.word_and_tag[sentence[0]].keys()):
                    dynamic_mat[t, 0] = self.tag_and_tag['start'][All_Tags[t]] * self.word_and_tag[sentence[0]][All_Tags[t]]
            else:
                if (All_Tags[t] in self.tag_and_tag['start'].keys()):
                    dynamic_mat[t, 0] = self.tag_and_tag['start'][All_Tags[t]] * 1   ### ignore unknown words

        ### update values from previous values of dynamic mat
        for w in range(1, len(sentence)):
            for t1 in range(0, len(All_Tags)):
                Max = 0
                tag = 0
                for t2 in range(0, len(All_Tags)):
                    if All_Tags[t2] in self.tag_and_tag[All_Tags[t1]].keys():
                        if dynamic_mat[t2, w-1] * self.tag_and_tag[All_Tags[t1]][All_Tags[t2]] > Max:
                            Max = dynamic_mat[t2, w-1] * self.tag_and_tag[All_Tags[t1]][All_Tags[t2]]
                            tag = t2
                if sentence[w] in self.word_and_tag.keys():
                    if All_Tags[t1] in self.word_and_tag[sentence[w]].keys():
                        dynamic_mat[t1][w] = Max * (self.word_and_tag[sentence[w]][All_Tags[t1]])
                        ## save best sequence
                        best_tag_sequence[t1][w] = int(tag)
                else:
                    dynamic_mat[t1][w] = Max * 1  ## ignore unknown words
                    best_tag_sequence[t1][w] = int(tag)

        end_Max = 0
        end_tag = 0
        for t1 in range(0, len(All_Tags)):
            if 'end' in self.tag_and_tag[All_Tags[t1]].keys():
                if dynamic_mat[t1, len(sentence) - 1] * self.tag_and_tag[All_Tags[t1]]['end'] > end_Max:
                    end_Max = dynamic_mat[t1, len(sentence) - 1] * self.tag_and_tag[All_Tags[t1]]['end']
                    end_tag = t1

        ######################
        #### BackTrace : ####
        predict_viterbi = []
        predict_viterbi.append(All_Tags[end_tag])
        this_tag = int(end_tag)
        for w in reversed(range(0, len(sentence))):
            try:
                predict_viterbi.append(All_Tags[int(best_tag_sequence[this_tag][w])])
                this_tag = int(best_tag_sequence[this_tag][w])
            except:
                print(this_tag)
                print(w)

        return predict_viterbi

    ### Find tags of all sentences of data with viterbi algorithm
    def tag_allsentences(self):

        sentence = []
        sent = 0

        with open('Data/POST-Persian-Corpus-Test-MyOut.txt', 'w') as file:
            for i in range(0, len(self.lines_test)-1):
                ### split sentences with '.'
                if self.lines_test[i][0] != '.':
                    if len(self.lines_test[i]) > 1:
                        sentence.append(self.lines_test[i][0])

                if self.lines_test[i][0] == '.':
                    sent += 1
                    if len(sentence) != 0:
                        self.All_sentences.append(sentence)
                        tag_predict = list(reversed(self.MyViterbi(sentence)))[1:]
                        self.tag_predicted.append(tag_predict)

                        for s in range(len(sentence)):
                            file.write(sentence[s] + '\t' + tag_predict[s] + '\n')
                        file.write('.\tDELM\n')

                    sentence = []

        # return self.tag_predicted