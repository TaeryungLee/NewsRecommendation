import csv
import os
from pprint import pprint

path = "nyt/comments/train"
filenames = os.listdir(path)
sums = {}

for fn in filenames:
    f = open("nyt/comments/train/"+fn, 'r', encoding='UTF8')
    rdr = csv.reader(f)
    data = []
    i = 0
    for line in rdr:
        if i==0:
            feat = line
            i=1
        else:
            data.append(line)

    comid = feat.index("commentID")
    ids = feat.index("userID")

    for dat in data:
        commentid = dat[comid]
        uid = dat[ids]
        if uid == '0':
            print(dat)
            print(fn)
            break
        if uid in sums.keys():
            sums[uid] += 1
        else:
            sums[uid] = 1

trainsums = sorted(sums.items(), key=lambda x: x[1], reverse=True)
traindict = {}

for t in trainsums:
    traindict[t[0]] = t[1]

path = "nyt/comments/test"
filenames = os.listdir(path)
sums = {}

for fn in filenames:
    f = open("nyt/comments/test/"+fn, 'r', encoding='UTF8')
    rdr = csv.reader(f)
    data = []
    i = 0
    for line in rdr:
        if i==0:
            feat = line
            i=1
        else:
            data.append(line)

    comid = feat.index("commentID")
    ids = feat.index("userID")

    for dat in data:
        commentid = dat[comid]
        uid = dat[ids]
        if uid == '0':
            print(dat)
            print(fn)
            break
        if uid in sums.keys():
            sums[uid] += 1
        else:
            sums[uid] = 1

testsums = sorted(sums.items(), key=lambda x: x[1], reverse=True)
testdict = {}

for t in testsums:
    testdict[t[0]] = t[1]

ids = list(testdict.keys()) + list(traindict.keys())
finid = []

for id in ids:
    if (id not in testdict.keys()) or (id not in traindict.keys()):
        continue
    if traindict[id] > 200 and testdict[id] > 100:
        finid.append(id)

fin = {}

for fi in finid:
    fin[fi] = (traindict[fi], testdict[fi])

pprint(fin)

print(len(fin))

f = open()


for id in finid[:5]:
    print(id)

for id in finid[:5]:
    print(id)