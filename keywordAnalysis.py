import re
import nltk

def keywordanalysis(userids, traincomments, articlekeywords):
    # Preference analysis
    # 1. Keyword based
    # Used data:
    # coOccurrence[keyword1][keyword2]: Co-occurrence between keyword1 and keyword2
    # userids: List of user id which we will focus on
    # traincomments[userid] = [(articleid, comment, commentid, userid), ...] : list of comments user commented
    # (Includes comment from users we are not interested in.)
    # articlekeywords[articleid]: keyword set of article

    # What to do: Extract user's interest in single comment
    # 개별 comment에 대해 그 코멘트가 얼마나 기사에 대한 관심을 표현하는지를 계산할 것
    # keyword는 word 단위로 쪼개서 lowercase로 정리되어있음 ("United States" => "united", "states")
    # 기사마다 주어진 keyword를 코멘트가 얼마나 포함하는가?
    # 예시: keyword는 "Donald", "Trump"
    # comment: "I like Donald Trump" => 전체 word 중 2개가 keyword와 겹치므로 관심이 있다고 볼 수 있다.
    # Output data format: Dictionary
    # keywordPreference[commentid] = 0~1사이의 정수값
    # **주의: list, dictionary 등 mutable data structure 변경하지 말 것

    # 먼저 keyword 포함 비율을 각 comment id에 대해 저장할 dictionary를 만든다.
    keywordpreference = {}
    # 우리가 관심있는 모든 userid에 대해 for문을 돌아 주며 비율을 계산한다.
    for user_id in userids:
        # traincomments 내에 각 user id를 key로 하는 list에 대해 여러 개의 comment tuple이 존재하므로 그에 대해 for문을 돌아준다.
        for comment_tuple in traincomments[user_id]:
            # 각 commment_tuple에는 0번이 article id이므로 이 정보를 저장해놓는다.
            article_id = comment_tuple[0]
            # 위에서 저장해 놓은 article id를 이용하여 이 article의 keyword를 articlekeyword에서 가져온다.
            article_keyword = articlekeywords[article_id]
            # comment id를 key로 하여 keywordpreference에 저장할 것이기 때문에 comment id를 comment tuple에서 가져와서 저장한다.
            comment_id = comment_tuple[2]
            # 그 뒤 comment tuple 내의 comment가 현재 list로 나누어져 있으므로
            # 각각의 token에 대해 그 token이 article_keyword에 들어있는 지를 확인하여 개수를 세준다.

            comment_tokens_raw = nltk.tokenize.word_tokenize(comment_tuple[1])
            comment_tokens = []
            for comment in comment_tokens_raw:
                comment_tokens.append(re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', comment))
            number_of_keywords = sum([1 for comment in comment_tokens if comment.lower() in article_keyword])
            # 그리고 comment의 token의 총 수를 저장한다.
            length_of_words = len(comment_tuple[1])
            # 마지막으로 comment_id를 key로 하여 keyword의 비율을 dictionary에 저장해준다.
            if length_of_words != 0:
                keywordpreference[comment_id] = number_of_keywords / (length_of_words ** 0.5)
            else:
                keywordpreference[comment_id] = 0

    return keywordpreference
