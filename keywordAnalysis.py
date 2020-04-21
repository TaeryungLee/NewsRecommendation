

def keywordanalysis(cooccurrence, userids, traincomments, articlekeywords):
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
    # comment는 word 단위로 쪼개서 lowercase로 정리되어있음 ("United States" => "united", "states")
    # 기사마다 주어진 keyword를 코멘트가 얼마나 포함하는가?
    # 예시: keyword는 "Donald", "Trump"
    # comment: "I like Donald Trump" => 전체 word 중 2개가 keyword와 겹치므로 관심이 있다고 볼 수 있다.
    # Output data format: Dictionary
    # keywordPreference[commentid] = 0~1사이의 정수값
    # **주의: list, dictionary 등 mutable data structure 변경하지 말 것

    keywordpreference = {}

    return keywordpreference

