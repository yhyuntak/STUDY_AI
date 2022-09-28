from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document): # 단어의 빈도수를 넣은 list를 생성하는 함수
  # 온점 제거 및 형태소 분석
  document = document.replace('.', '')
  tokenized_document = okt.morphs(document) # 형태소로 토큰화

  word_to_index = {}
  bow = []

  for word in tokenized_document:
    if word not in word_to_index.keys(): # dictionary에 있지 않으면,
      word_to_index[word] = len(word_to_index)
      # BoW에 전부 기본값 1을 넣는다.
      bow.insert(len(word_to_index) - 1, 1) #
    else:
      # 재등장하는 단어가 있으면 인덱스를 추출
      index = word_to_index.get(word)
      # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
      bow[index] = bow[index] + 1

  return word_to_index, bow

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
print('vocabulary :', vocab)
print('bag of words vector :', bow)


from sklearn.feature_extraction.text import CountVectorizer
text = ["Family is not an important thing. It's everything."]
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
print("BoW vector :",vector.fit_transform(text).toarray())

# 각 단어의 인덱스 확인
print("idx :",vector.vocabulary_)


from sklearn.feature_extraction.text import CountVectorizer
text = ["Family is not an important thing. It's everything."]
vector = CountVectorizer(stop_words=['the','a','an','is','not'])

# 코퍼스로부터 각 단어의 빈도수를 기록
print("BoW vector :",vector.fit_transform(text).toarray())

# 각 단어의 인덱스 확인
print("idx :",vector.vocabulary_)

