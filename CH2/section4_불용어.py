"""
갖고 있는 데이터에서 유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요합니다.

예를 들면, I, my, me, over, 조사, 접미사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없는 경우가 있습니다.
이러한 단어들을 불용어(stopword)라고 합니다.

"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

stop_words_list = stopwords.words('english')
print('불용어 개수 :', len(stop_words_list))
print('불용어 10개 출력 :',stop_words_list[:10])

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

# 단어 단위로 토큰화된걸
word_tokens = word_tokenize(example)

# 하나씩 뽑아서 불용어에 속하는지 확인하고
results = []
for word in word_tokens :
    if word not in stop_words :
        # 속하지 않으면 추가
        results.append(word)

print('불용어 제거 전 :',word_tokens)
print('불용어 제거 후 :',results)

