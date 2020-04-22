

def sentimentanalysis(userids, traincomments):
    # userids: List of user id which we will focus on
    # traincomments[userid] = [(articleid, comment, commentid, userid), ...] : list of comments user commented
    # (Includes comment from users we are not interested in.)

    # What to do: Extract user's interest in single comment based on sentiment polarity

    # Output data format: Dictionary
    # sentimentpreference[commentid] = (pos: 0~1사이의 정수값, neu: 0~1사이의 정수값, neg: 0~1사이의 정수값) (softmax)
    # **주의: list, dictionary 등 mutable data structure 변경하지 말 것

    sentimentpreference = {}
    for userid in userids:
        for comment in traincomments[userid]:
            sentimentpreference[comment[2]] = {'pos': 1, 'neu': 0, 'neg': 0}

    return sentimentpreference
