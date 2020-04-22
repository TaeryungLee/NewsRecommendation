import io
import numpy
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api
import elice_utils_emotion
import re
import math
import os

def main():
    pos_sentences = read_text_data('./txt_sentoken/pos/')
    neg_sentences = read_text_data('./txt_sentoken/neg/')
    testing_sentence = input("Input Sentence > ")

    ###############
    #Do not Change#
    ###############
    alpha = 0.1
    prob1 = 0.5 #Prior of POSITIVE
    prob2 = 0.5 #Prior of NEGATIVE

    prob_pair = naive_bayes(pos_sentences, neg_sentences, testing_sentence, alpha, prob1, prob2)

    print("P(pos|sentence) : ", prob_pair[0])
    print("P(neg|sentence) : ", prob_pair[1])

def naive_bayes(pos_sentence, neg_sentence, testing_sentence, alpha, prob1, prob2):
    '''
    By using Naive Bayes Classifier
    Calculate log P(class1|test), log P(class2|test)
    return normalized probability.

    prob1 == P(class1)
    prob2 == P(class2)
    Be careful prob1, prob2 are not log probabilities.

    But, to make assignment easy, it is already implemented.
    '''

    pos_model = create_BOW(pos_sentence)
    neg_model = create_BOW(neg_sentence)
    testing_model = create_BOW(testing_sentence)

    pos_prob = calculate_doc_prob(pos_model, testing_model, alpha) + math.log(prob1)
    neg_prob = calculate_doc_prob(neg_model, testing_model, alpha) + math.log(prob2)

    return normalize_log_prob(pos_prob, neg_prob)

def read_text_data(directory):
    '''
    This function is already implemented.abs
    '''

    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.txt')]

    all_text = ''
    for f in files:
        all_text += ' '.join(open(directory + f).readlines()) + ' '

    return all_text

def normalize_log_prob(logprob1, logprob2):
    '''
    logprob1, logprob2 are log probabilities.abs    By normalizing these, return 2 probabilities that satisfy normalized_prob1 + normalized_prob2 = 1.
    You can transform log distribution to linear distribution by using math.exp().


    In this step, keep in mind if both log probabilities is very small, it can be 0 when it transformed to linear distribution.
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

    #Your code here
    addee = max(logprob1, logprob2)
    #print(logprob1)
    #print(logprob2)
    #print(addee)
    p1 = math.exp(logprob1-addee)
    p2 = math.exp(logprob2-addee)
    normalized_prob1 = p1/(p1+p2)
    normalized_prob2 = p2/(p1+p2)

    return (normalized_prob1, normalized_prob2)

def calculate_doc_prob(training_bow, testing_bow, alpha):
    logprob = 0

    '''
    return log probabilities after smoothing using alpha.abs    '''
    #Your code here
    tr_bow_dict, tr_bow = training_bow
    ts_bow_dict, ts_bow = testing_bow
    #smoothing
    bot = 0
    length = 0
    for i in tr_bow.keys():
        bot += tr_bow[i]
        length += 1
    den = bot + (alpha*length)

    p=[]
    for i in tr_bow.keys():     #training.idx2count로부터 training.idx2probability dictionary 생성
        elem = tr_bow[i] + alpha
        elem/=den
        p.append((i, elem))
    p_dict = dict(p)
    #print(p_dict)
    for w in ts_bow_dict.keys():     #testing.word2idx에서 word마다
        if w not in tr_bow_dict.keys():     #training.word2idx에서 학습되지 않았다면 특정 값 부여
            logprob += math.log(alpha/den)
        else:
            logprob += (ts_bow[ts_bow_dict[w]])*math.log(p_dict[tr_bow_dict[w]])     #add log-probability using p_dict

    return logprob

def create_BOW(sentence):
    bow_dict = {} #word to index
    bow_ = [] # index to count

    '''
    function that makes Bag of Words.abs
    '''
    #Your code here
    lst = sentence.split('\n')
    dummy = ''
    for i in range(len(lst)):
        dummy += replace_non_alphabetic_chars_to_space(lst[i].lower())
        dummy += ' '
    Bag = dummy.split()
    bag_set = sorted(set(Bag))
    Bag = sorted(Bag)
    bow_dict = dict([(Bag[i], i) for i in range(len(Bag))])
    j = 0
    i = 0
    for j in range(len(bag_set)):
        count = 0
        while Bag[i] == bag_set[j]:
            count+=1
            i+=1
            if i == len(Bag):
                break
        bow_.append((bow_dict[bag_set[j]], count))
    bow = dict(bow_)

    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)


if __name__ == "__main__":
    main()

