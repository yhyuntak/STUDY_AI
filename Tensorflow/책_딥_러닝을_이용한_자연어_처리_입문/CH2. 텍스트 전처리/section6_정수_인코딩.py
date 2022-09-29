from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
sentences = sent_tokenize(raw_text)
print(sentences)

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences :
    # 단어로 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence :
        word= word.lower() # 소문자화
        if word not in stop_words: #불용어 제거
            if len(word) > 2 : # 길이가 2 이하인 것들 제거
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word]+=1
    preprocessed_sentences.append(result)
print(preprocessed_sentences)
print("단어 집합:",vocab)

vocab_sorted = sorted(vocab.items(),key=lambda x : x[1],reverse= True)
print(vocab_sorted)

word_to_index = {}
i = 0
for (word,frequency) in vocab_sorted :
    if frequency > 1 : #빈도수가 작은 단어는 제외
        i += 1
        word_to_index[word] = i
print(word_to_index)

vocab_size = 5

# 인덱스가 6 이상인 단어들만 추가.
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

# 해당 단어에 대한 인덱스 정보를 삭제
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)

word_to_index['OOV'] = len(word_to_index)+1

encoded_sentences = []
for sentence in preprocessed_sentences :
    encoded_sentence = []
    for word in sentence :
        try :
            encoded_sentence.append(word_to_index[word])
        except KeyError :
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)

from collections import Counter

all_words_list = sum(preprocessed_sentences,[])
print(all_words_list)
vocab = Counter(all_words_list)
print(vocab)

vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)

word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i

print(word_to_index)

from nltk import FreqDist
import numpy as np
vocab = FreqDist(np.hstack(preprocessed_sentences))
print(vocab['barber'])

from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size +2 ,oov_token = "OOV")
# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences)
print(tokenizer.word_index)
print(tokenizer.word_counts)
# 토크나이저 객체를 정제,정규화되고 단어 토큰화된 문장을 학습해서 문자열을 정수화
print(tokenizer.texts_to_sequences(preprocessed_sentences))

