import nltk
from nltk.corpus import stopwords
import os
import csv
import re
import datetime
import gzip
from numpy import dot
from numpy.linalg import norm
import pickle
import multiprocessing
import parmap
from keywordAnalysis import keywordanalysis
from pprint import pprint
from sentimentAnalysis import sentimentanalysis
from keywordAnalysis2 import keywordanalysis2

"""
News Recommendation based on Comment Preference Analysis

KAIST, CS372: Natural Language Processing using Python

Team18
Howook Shin
Taeryung Lee
Hyunki Lee
Youchan Park
Junwoo Choi
"""
if __name__ == "__main__":
    # Starting time record
    start = datetime.datetime.now()
    print("Starting time: ", start)

    # Global features used
    stopword = list(stopwords.words('english'))

    # <Load dataset>
    # Data: https://www.kaggle.com/aashita/nyt-comments
    # Data location in "nyt/articles(comments)/train(test)/Articles(comments)MonthYear.csv"
    # Article data format
    # Csv column index: [articleID, articleWordCount, byline, documentType, headline, keywords, multimedia, newDesk,
    #                    printPage, pubDate, sectionName, snippet, source, typeOfMaterial, webURL]
    # Index we use: [articleID, articleWordCount, keywords, headline, pubDate]

    # Comment data format
    # Csv column index: [approveDate, articleID, articleWordCount, commentBody, commentID, commentSequence, commentTitle,
    #                    commentType, createDate, depth, editorsSelection, inReplyTo, newDesk, parentID,
    #                    parentUserDisplayName, permID, picURL, printPage, recommendations, recommendedFlag, replyCount,
    #                    reportAbuseFlag, sectionName, sharing, status, timespeople, trusted, typeOfMaterial, updateDate,
    #                    userDisplayName,userID,userLocation,userTitle,userURL]
    # Index we use: [articleID, commentBody, commentID, userID]
    #
    # Raw data format
    # comments: {userid: [(articleid, comment, commentid, userid), (articleid, comment, commentid, userid), ...] ....}
    # articles: [(articleid, headline, wordcount, keywords, publishdate), ....]
    #
    # Crawled article body
    # nyt/articles/body/(articleID).txt

    traincomments = {}
    trainarticles = []
    testcomments = {}
    testarticles = []
    alluserids = []
    traincommentcounts = {}
    testcommentcounts = {}

    path = "nyt/comments/train"
    filenames = os.listdir(path)

    for fn in filenames:
        f = open("nyt/comments/train/" + fn, 'r', encoding='UTF8')
        rdr = csv.reader(f)
        data = []
        i = 0
        for line in rdr:
            if i == 0:
                feat = line
                i = 1
            else:
                data.append(line)
        artid = feat.index("articleID")
        combd = feat.index("commentBody")
        comid = feat.index("commentID")
        ids = feat.index("userID")
        for dat in data:
            commentid = dat[comid]
            uid = dat[ids]
            articleid = dat[artid]
            commentbody = dat[combd]
            if uid not in traincomments.keys():
                traincomments[uid] = [(articleid, commentbody, commentid, uid)]
                traincommentcounts[uid] = 1
            else:
                traincomments[uid].append((articleid, commentbody, commentid, uid))
                traincommentcounts[uid] += 1

    path = "nyt/comments/test"
    filenames = os.listdir(path)

    for fn in filenames:
        f = open("nyt/comments/test/" + fn, 'r', encoding='UTF8')
        rdr = csv.reader(f)
        data = []
        i = 0
        for line in rdr:
            if i == 0:
                feat = line
                i = 1
            else:
                data.append(line)
        artid = feat.index("articleID")
        combd = feat.index("commentBody")
        comid = feat.index("commentID")
        ids = feat.index("userID")
        for dat in data:
            commentid = dat[comid]
            uid = dat[ids]
            articleid = dat[artid]
            commentbody = dat[combd]
            if uid not in testcomments.keys():
                testcomments[uid] = [(articleid, commentbody, commentid, uid)]
                testcommentcounts[uid] = 1
            else:
                testcomments[uid].append((articleid, commentbody, commentid, uid))
                testcommentcounts[uid] += 1

    path = "nyt/articles/train"
    filenames = os.listdir(path)

    for fn in filenames:
        f = open("nyt/articles/train/" + fn, 'r', encoding='UTF8')
        rdr = csv.reader(f)
        data = []
        i = 0
        for line in rdr:
            if i == 0:
                feat = line
                i = 1
            else:
                data.append(line)
        artid = feat.index("articleID")
        countid = feat.index("articleWordCount")
        kwdid = feat.index("keywords")
        hdid = feat.index("headline")
        pubid = feat.index("pubDate")
        for dat in data:
            articleid = dat[artid]
            wordcount = dat[countid]
            headline = dat[hdid]
            pubdate = dat[pubid]
            rawkeywords = dat[kwdid]
            reg = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', rawkeywords).split(" ")
            keywords = []
            for keyword in reg:
                if keyword not in stopword and len(keyword) > 3:
                    keywords.append(keyword)
            trainarticles.append((articleid, headline, wordcount, keywords, pubdate))

    path = "nyt/articles/test"
    filenames = os.listdir(path)

    for fn in filenames:
        f = open("nyt/articles/test/" + fn, 'r', encoding='UTF8')
        rdr = csv.reader(f)
        data = []
        i = 0
        for line in rdr:
            if i == 0:
                feat = line
                i = 1
            else:
                data.append(line)
        artid = feat.index("articleID")
        countid = feat.index("articleWordCount")
        kwdid = feat.index("keywords")
        hdid = feat.index("headline")
        pubid = feat.index("pubDate")
        for dat in data:
            articleid = dat[artid]
            wordcount = dat[countid]
            headline = dat[hdid]
            pubdate = dat[pubid]
            rawkeywords = dat[kwdid]
            reg = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', rawkeywords).split(" ")
            keywords = []
            for keyword in reg:
                if keyword not in stopword and len(keyword) > 3:
                    keywords.append(keyword)
            testarticles.append((articleid, headline, wordcount, keywords, pubdate))
    # Loading done, record time
    loadtime = datetime.datetime.now()
    print("Loading done. Loading time: ", loadtime - start)

    # Preprocessing
    # Two steps:
    # 1. Comment preprocessing
    # 1.1 Organize comment by users using user id (Done with data loading)
    # traincomments, testcomments have userid as keys, set of comments as values
    # traincommentcounts, testcommentcounts have userid as keys, number of comment as values

    alluserids = list(traincomments.keys()) + list(testcomments.keys())

    # 1.2 Filter 'hard commenters' who commented more than 100 articles in both training set and test set

    """
    for userid in alluserids:
        if userid not in list(traincomments.keys()) or userid not in list(testcomments.keys()):
            continue
        if traincommentcounts[userid] >= 100 and testcommentcounts[userid] >= 100:
            userids.append(userid)
    """

    # Target user ids saved in userid.txt due to long runtime
    with open("userid.txt", 'r') as f:
        userids = f.read().split("\n")
    rawtargetTrainComments = {}
    rawtargetTestComments = {}
    targetTrainComments = {}
    targetTestComments = {}
    for userid in userids:
        rawtargetTrainComments[userid] = traincomments[userid]
        rawtargetTestComments[userid] = testcomments[userid]

    # 2. Keyword preprocessing
    # 2.1 Form a set of keywords
    keywordset = []
    articlekeywords = {}
    taggedarticlekeywords = {}

    if os.path.isfile("tagged.pickle"):
        with gzip.open("tagged.pickle", "rb") as f:
            taggedarticlekeywords = pickle.load(f)
        taggedloaded = True
    else:
        taggedloaded = False

    for article in trainarticles:
        keywords = article[3]
        articleid = article[0]
        articlekeywords[articleid] = [keyword.lower() for keyword in keywords]
        if not taggedloaded:
            taggedarticlekeywords[articleid] = [nltk.pos_tag([keyword.lower()])[0] for keyword in keywords]
        for keyword in keywords:
            if keyword not in keywordset:
                keywordset.append(keyword.lower())

    if not taggedloaded:
        with gzip.open("tagged.pickle", "wb") as f:
            pickle.dump(taggedarticlekeywords, f)

    for article in testarticles:
        keywords = article[3]
        articleid = article[0]
        articlekeywords[articleid] = [keyword.lower() for keyword in keywords]
        for keyword in keywords:
            if keyword not in keywordset:
                keywordset.append(keyword.lower())

    for userid in userids:
        targetTrainComments[userid] = []
        targetTestComments[userid] = []
        for comment in rawtargetTrainComments[userid]:
            if comment[0] in articlekeywords.keys():
                targetTrainComments[userid].append(comment)
        for comment in rawtargetTestComments[userid]:
            if comment[0] in articlekeywords.keys():
                targetTestComments[userid].append(comment)

    # 2.2 Count co-occurrence in keyword set

    coOccurrence = {}
    for article in (trainarticles + testarticles):
        articleid = article[0]
        keywords = article[3]
        articlekeywords[articleid] = keywords.copy()
        for keyword in keywords:
            if keyword not in coOccurrence.keys():
                coOccurrence[keyword] = {}
            for keyword2 in keywords:
                if keyword2 not in coOccurrence[keyword].keys():
                    coOccurrence[keyword][keyword2] = 1
                else:
                    coOccurrence[keyword][keyword2] += 1

    # Preprocessing done
    preptime = datetime.datetime.now()
    print("Preprocessing done. Preprocessing time: ", preptime - loadtime)

    # Preference analysis
    # 1. Keywords based
    # Used data:
    # coOccurrence[keyword1][keyword2]: Co-occurrence between keyword1 and keyword2
    # userids: List of user id
    # traincomments[userid] = [(articleid, comment, commentid, userid), ...] : list of comments user commented
    # articlekeywords[articleid]: keyword set of article

    # What to do: Extract user's interest in single comment based on keywords
    # 개별 comment에 대해 그 코멘트가 얼마나 기사에 대한 관심을 표현하는지를 계산할 것
    # comment는 word 단위로 쪼개서 lowercase로 정리되어있음 ("United States" => "united", "states")
    # 기사마다 주어진 keyword를 코멘트가 얼마나 포함하는가?
    # 예시: keyword는 "Donald", "Trump"
    # comment: "I like Donald Trump" => 전체 word 중 2개가 keyword와 겹치므로 관심이 있다고 볼 수 있다.
    # Output data format: Dictionary
    # keywordpreference[commentid] = 0~1사이의 정수값

    # keywordPreference = keywordanalysis(userids, targetTrainComments, taggedarticlekeywords)

    # Multiprocessing code, 10 threads.
    # Takes 2 hours in i7-8700 cpu
    if os.path.isfile("kwdpref.pickle"):
        with gzip.open("kwdpref.pickle", "rb") as f:
            keywordPreference = pickle.load(f)
    else:
        # Change threads by your environment
        threads = 10
        splitids = [round(i * len(userids) / threads) for i in range(threads + 1)]
        useridlst = []
        for i in range(threads):
            useridlst.append([tuple(userids[splitids[i]:splitids[i + 1]])])

        result = parmap.starmap(keywordanalysis2, useridlst, targetTrainComments, taggedarticlekeywords)
        keywordPreference = {}
        for res in result:
            keywordPreference.update(res)

        with gzip.open("kwdpref.pickle", "wb") as f:
            pickle.dump(keywordPreference, f)

    """
    # Single Thread code
    keywordPreference = keywordanalysis2(userids, targetTrainComments, taggedarticlekeywords)
    """
    # Normalize
    sums = 0
    count = 0
    for comid in keywordPreference.keys():
        count += 1
        sums += keywordPreference[comid]
    average = sums / count
    for comid in keywordPreference.keys():
        keywordPreference[comid] = keywordPreference[comid] / average

    # 2. Sentiment polarity based
    # Used data:
    # userids: List of user id
    # traincomments[userid] = [(articleid, comment, commentid, userid), ...] : list of comments user commented

    # What to do: Extract user's interest in single comment based on sentiment polarity

    # Output data format: Dictionary
    # sentimentpreference[commentid] = (pos: 0~1사이의 정수값, neu: 0~1사이의 정수값, neg: 0~1사이의 정수값) (softmax)

    if os.path.isfile("sentpref.pickle"):
        with gzip.open("sentpref.pickle", "rb") as f:
            sentimentPreference = pickle.load(f)
    else:
        sentimentPreference = sentimentanalysis(userids, targetTrainComments)
        with gzip.open("sentpref.pickle", "wb") as f:
            pickle.dump(sentimentPreference, f)

    # Keyword, Sentiment analysis done
    preftime = datetime.datetime.now()
    print("Keyword, Sentiment analysis done. Analysis time: ", preftime - preptime)

    # 3. Preference on keywords
    # keyword base: 기본점수 1점, keyword 연관성 0~2점
    # sentiment base: pos 2점, neg 1점, neu 0점
    userPrefKeyword = {}

    # Best hyperparameter searched
    ratio = 0.41
    for userid in userids:
        userPrefKeyword[userid] = {}
        for comment in targetTrainComments[userid]:
            articleid = comment[0]
            commentid = comment[2]
            keywords = articlekeywords[articleid]
            keywordpref = keywordPreference[commentid]
            sentimentpref = sentimentPreference[commentid]

            sentimentPoint = 0
            if sentimentpref['pos'] > 0.8 or sentimentpref['pos'] < 0.2:
                sentimentPoint = max(sentimentpref['pos'], sentimentpref['neg'])

            # Fitted here
            preference = (ratio * keywordpref) + ((1 - ratio) * sentimentPoint)

            for keyword in keywords:
                if keyword not in userPrefKeyword[userid].keys():
                    userPrefKeyword[userid][keyword] = preference
                else:
                    userPrefKeyword[userid][keyword] += preference

        # Softmax, so we have a softmax preference vector on each user(sum of vector elements is 1)
        sums = 0
        for keyword in userPrefKeyword[userid].keys():
            sums += userPrefKeyword[userid][keyword]

        for keyword in userPrefKeyword[userid].keys():
            userPrefKeyword[userid][keyword] = userPrefKeyword[userid][keyword] / sums

    # Keyword preference analysis done
    kwdtime = datetime.datetime.now()
    print("Keyword preference analysis done. Analysis time: ", kwdtime - preftime)

    # User preference on unseen articles
    userPrefArticle = {}
    for userid in userids:
        userPrefArticle[userid] = {}
        for article in testarticles:
            articleid = article[0]
            prefsum = 0
            for keyword in articlekeywords[articleid]:
                if keyword in userPrefKeyword[userid].keys():
                    prefsum += userPrefKeyword[userid][keyword] / len(articlekeywords[articleid])
            userPrefArticle[userid][articleid] = prefsum

    # Article preference analysis done
    articletime = datetime.datetime.now()
    print("Article preference analysis done. Analysis time: ", articletime - kwdtime)

    # News recommendation based on preference
    recommend = {}
    for userid in userids:
        pref = userPrefArticle[userid]
        prefsort = sorted(pref.items(), key=lambda x: x[1], reverse=True)
        recommend[userid] = [item[0] for item in prefsort[:150]]

    # Recommendation done
    rectime = datetime.datetime.now()
    print("Recommendation done. Recommendation time: ", rectime - articletime)

    # Ground truth
    groundtruth = {}
    for userid in userids:
        groundtruth[userid] = []
        for comment in targetTestComments[userid]:
            groundtruth[userid].append(comment[0])

    # Recommendation based on user clustering
    # First, calculate cosine similarity of users based on preference vector
    # Preference information is contained in userPrefKeyword[userid][keyword]
    # keyword set is contained in keywordset as list
    # Hyperparameter is a threshold, how large should be cosine similarity to consider two vectors are similar

    similarity = {}
    simvec = {}
    # Form similarity vector
    for userid in userids:
        vec = []
        for keyword in keywordset:
            if keyword in userPrefKeyword[userid].keys():
                vec.append(userPrefKeyword[userid][keyword])
            else:
                vec.append(0)
        simvec[userid] = vec

    if os.path.isfile("similarity.pickle"):
        with gzip.open("similarity.pickle", "rb") as f:
            similarity = pickle.load(f)
    else:
        i = 0
        for userid1 in userids:
            user_sim = {}
            i += 1
            print(i, len(userids))
            for userid2 in userids:
                if userid1 == userid2:
                    continue
                # compute cosine similarity
                vec1 = simvec[userid1]
                vec2 = simvec[userid2]
                cossim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
                if userid1 not in similarity.keys():
                    similarity[userid1] = {userid2: cossim}
                else:
                    similarity[userid1][userid2] = cossim
        with gzip.open("similarity.pickle", "wb") as f:
            pickle.dump(similarity, f)

    # cosine similarity computed
    nbds = {}
    # For each user, find which users have similar preference
    for userid1 in userids:
        for userid2 in userids:
            best = 0
            uid = 0
            if userid1 == userid2:
                continue
            if best < similarity[userid1][userid2]:
                best = similarity[userid1][userid2]
                uid = userid2
            nbds[userid1] = userid2

    # neighborhood users found

    # recommendation based on clustering
    # list up articles that neighbors have commented
    cluster_recommend = {}

    for userid in userids:
        nbd = nbds[userid]
        if len(nbds[userid]) == 0:
            cluster_recommend[userid] = recommend[userid]
        else:
            cluster_recommend[userid] = groundtruth[nbd]
    # Hybrid recommendation
    # Mixed method of preference and clustering
    # Priorly use clustering based method, and add up until 100 recommendation from preference based recommendation

    hybrid_recommend = {}

    for userid in userids:
        hybrid_recommend[userid] = [artid for artid in cluster_recommend[userid]][:75]
        i = 0
        while len(hybrid_recommend[userid]) < 150:
            if recommend[userid][i] not in hybrid_recommend[userid]:
                hybrid_recommend[userid].append(recommend[userid][i])
            i += 1

    # Hybrid recommendation done

    # recommendation done
    rectime2 = datetime.datetime.now()
    print("Recommendation done. Recommendation time: ", rectime2 - rectime)

    # Evaluation for preference based recommendation
    tpsum = 0
    fpsum = 0
    fnsum = 0

    for userid in userids:
        acc = 0
        for recom in recommend[userid]:
            if recom in groundtruth[userid]:
                acc += 1
        tp = acc
        fp = len(recommend[userid]) - acc
        fn = len(testarticles) - tp - fp - len(groundtruth[userid])
        tpsum += tp
        fpsum += fp
        fnsum += fn
    prefprecision = tpsum / (tpsum + fpsum)
    prefrecall = tpsum / (tpsum + fnsum)
    preffscore = 2 * prefprecision * prefrecall / (prefprecision + prefrecall)
    print("preference based recommendation")
    print("Precision: ", prefprecision, "Recall", prefrecall, "F-score", preffscore)

    # Evaluation for clustering based recommendation
    tpsum = 0
    fpsum = 0
    fnsum = 0
    for userid in userids:
        acc = 0
        for recom in cluster_recommend[userid]:
            if recom in groundtruth[userid]:
                acc += 1
        tp = acc
        fp = len(cluster_recommend[userid]) - acc
        fn = len(testarticles) - tp - fp - len(groundtruth[userid])
        tpsum += tp
        fpsum += fp
        fnsum += fn

    clprecision = tpsum / (tpsum + fpsum)
    clrecall = tpsum / (tpsum + fnsum)
    clfscore = 2 * clprecision * clrecall / (clprecision + clrecall)
    print("clustering based recommendation")
    print("Precision: ", clprecision, "Recall", clrecall, "F-score", clfscore)

    # Evaluation for hybrid recommendation
    tpsum = 0
    fpsum = 0
    fnsum = 0
    for userid in userids:
        acc = 0
        for recom in hybrid_recommend[userid]:
            if recom in groundtruth[userid]:
                acc += 1
        tp = acc
        fp = len(hybrid_recommend[userid]) - acc
        fn = len(testarticles) - tp - fp - len(groundtruth[userid])
        tpsum += tp
        fpsum += fp
        fnsum += fn

    hybprecision = tpsum / (tpsum + fpsum)
    hybrecall = tpsum / (tpsum + fnsum)
    hybfscore = 2 * hybprecision * hybrecall / (hybprecision + hybrecall)
    print("hybrid based recommendation")
    print("Precision: ", hybprecision, "Recall", hybrecall, "F-score", hybfscore)
