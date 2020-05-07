import io
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import re
import math
import os
from nltk.corpus import stopwords as stp

class Prob_Bag():
    def __init__(self, testing_ment, Bags=None, pos_sents=None, neg_sents=None, pos_model = None, neg_model = None, alpha=0.1, prob_P=0.5, prob_N=0.5):
        self.test = testing_ment
        self.Bags = Bags or {}
        self.pos_sents = None
        self.neg_sents = None

        if pos_sents != None:
            if pos_sents.endswith('/') and pos_sents.startswith('/'):
                self.pos_sents = self.read_text_data(pos_sents)
            else:
                self.pos_sents = pos_sents
        elif pos_model == None:
            self.pos_sents = self.read_text_data('./txt_sentoken/pos/')
        if neg_sents != None:
            if neg_sents.endswith('/') and neg_sents.startswith('/'):
                self.neg_sents = self.read_text_data(neg_sents)
            else:
                self.neg_sents = neg_sents
        elif neg_model == None:
            self.neg_sents = self.read_text_data('./txt_sentoken/neg/')

        self.alpha = alpha
        self.prob_P = prob_P
        self.prob_N = prob_N
        self.pos_model = pos_model or ({},{})
        self.neg_model = neg_model or ({},{})
        self.pos_prob = math.log(0.5)
        self.neg_prob = math.log(0.5)


    def naive_bayes(self):
        '''
        By using Naive Bayes Classifier
        Calculate log P(class1|test), log P(class2|test)
        return normalized probability.abs 34
        prob1 == P(class1)
        prob2 == P(class2)
        Be careful prob1, prob2 are not log probabilities.
        
        But, to make assignment easy, it is already implemented.
        '''
        if self.pos_sents != None:
            print("creating pos model")
            self.pos_model = self.create_BOW(self.pos_sents, self.pos_model)
            print("done")
        if self.neg_sents != None:
            print("creating neg model")
            self.neg_model = self.create_BOW(self.neg_sents, self.neg_model)
            print("done")
        testing_model = self.create_BOW(self.test, ({},{}))

        self.pos_prob = self.calculate_doc_prob(self.pos_model, testing_model) + math.log(self.prob_P)
        self.neg_prob = self.calculate_doc_prob(self.neg_model, testing_model) + math.log(self.prob_N)

        return self.normalize_log_prob(self.pos_prob, self.neg_prob)

    def read_text_data(self, directory):
        '''
        This function is already implemented.abs 54
        '''

        files = os.listdir(directory)
        files = [f for f in files if f.endswith('.txt')]

        all_text = ''
        for f in files:
            all_text += ' '.join(open(directory + f).readlines()) + ' '

        return all_text

    def normalize_log_prob(self, logprob1, logprob2):
        '''
        logprob1, logprob2 are log probabilities.abs    By normalizing these, return 2 probabilities that satisfy nor
        You can transform log distribution to linear distribution by using math.exp().
        
        In this step, keep in mind if both log probabilities is very small, it can be 0 when it transformed to linear
        So, you have to do something before transforming.

        Example)
        logprob1 = -298
        logprob2 = -300
        In this case, it will be 0 when we use exp(). (underflow)
        So, we should add 298 to each probability.

        logprob1 = 0
        logprob2 = -2
        
        Then we apply exp(),

        prob1 = 1
        prob2 = 0.1353
        and normalizing these probabilities, we can finally get

        normalized_prob1 = 0.8808
        normalized_prob2 = 0.1192
        '''

        addee = max(logprob1, logprob2)
        p1 = math.exp(logprob1-addee)
        p2 = math.exp(logprob2-addee)
        normalized_prob1 = p1/(p1+p2)
        normalized_prob2 = p2/(p1+p2)

        return (normalized_prob1, normalized_prob2)

    def calculate_doc_prob(self, training_bow, testing_bow):
        logprob = 0
        '''
        return log probabilities after smoothing using alpha.abs
        '''

        tr_bow_dict, tr_bow = training_bow
        ts_bow_dict, ts_bow = testing_bow

        #smoothing
        bot = 0
        length = 0
        for i in tr_bow.keys():
            bot += tr_bow[i]
            length += 1
        den = bot + (self.alpha*length)

        p=[]
        for i in tr_bow.keys():
            #training.idx2count로부터 training.idx2probability dictionary 생성
            elem = tr_bow[i] + self.alpha
            elem/=den
            p.append((i, elem))
        p_dict = dict(p)
        
        for w in ts_bow_dict.keys():
            #testing.word2idx에서 word마다
            if w not in tr_bow_dict.keys():
                #training.word2idx에서 학습되지 않았다면 특정 값 부여
                logprob += math.log(self.alpha/den)
            else:
                logprob += (ts_bow[ts_bow_dict[w]])*math.log(p_dict[tr_bow_dict[w]])     #add log-probability using p132
        return logprob

    def create_BOW(self, sentence, model):
        (bow_dict, bow) = model
        if sentence==None:
            return model
        #bow_dict = {} #word to index
        #bow_=[] # index to count

        lastval = len(bow_dict.values())
        '''
        function that makes Bag of Words.abs
        '''
        lst = sentence.split('\n')
        dummy = ''
        for i in range(len(lst)):
            dummy += self.replace_non_alphabetic_chars_to_space(lst[i].lower())
            dummy += ' '
        Bag = dummy.split()
        for word in Bag:
            if word in stp.words('english'):
                Bag.remove(word)
        bag_set = sorted(set(Bag))
        Bag = sorted(Bag)

        for i in range(len(bag_set)):
            if bag_set[i] not in bow_dict.keys():
                bow_dict[bag_set[i]] = i+lastval

        i = 0
        for j in range(len(bag_set)):
            wd = bag_set[j]
            count = 0
            while Bag[i] == wd:
                count+=1
                i+=1
                if i == len(Bag):
                    break

            if bow_dict[wd] in bow.keys():
                val = bow[bow_dict[wd]]
                bow[bow_dict[wd]] = val+count
            else:
                bow[bow_dict[wd]] = count
        return (bow_dict, bow)

    def replace_non_alphabetic_chars_to_space(self, sentence):
        return re.sub(r'[^a-z]+', ' ', sentence)


def sentimentanalysis(userids, traincomments):
    # userids: List of user id which we will focus on
    # traincomments[userid] = [(articleid, comment, commentid, userid), ...] : list of comments user commented
    # (Includes comment from users we are not interested in.)

    # What to do: Extract user's interest in single comment based on sentiment polarity

    # Output data format: Dictionary
    # sentimentpreference[commentid] = (pos: 0~1사이의 정수값, neu: 0~1사이의 정수값, neg: 0~1사이의 정수값) (softmax)
    # **주의: list, dictionary 등 mutable data structure 변경하지 말 것

    sentimentpreference = {}
    p_mod = None
    n_mod = None
    i = 0
    total = 0
    for userid in userids:
        total += len(traincomments[userid])
    for userid in userids:
        for comment in traincomments[userid]:
            obj = Prob_Bag(comment[1], pos_model = p_mod, neg_model = n_mod)
            (pos_p, neg_p) = obj.naive_bayes()
            if i == 0:
                p_mod = obj.pos_model
                n_mod = obj.neg_model
            sentimentpreference[comment[2]] = {'pos': pos_p, 'neg': neg_p}
            if i % 1000 == 0:
                print(i, total, round(i/total*100, 2))
            i += 1

    return sentimentpreference

