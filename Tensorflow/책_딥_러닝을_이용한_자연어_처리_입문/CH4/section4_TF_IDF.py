import pandas as pd
from math import log

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
vocab = list(set(w for doc in docs for w in doc.split())) # 중복처리하는 방법이네 신기하다.
print(vocab)

N = len(docs)
def tf(term,doc):
    # 문장에서 단어가 몇번 나왔는지 리턴
    return doc.count(term)

def idf(t):
    df = 0
    for doc in docs :
        df += t in doc # 단어(t)가 문장에 있으면 df에 더한다.
    return log(N/(df+1))

def tfidf(t,d):
    return tf(t,d)*idf(t)

result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
  result.append([])
  d = docs[i]
  for j in range(len(vocab)):
    t = vocab[j]
    result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)


