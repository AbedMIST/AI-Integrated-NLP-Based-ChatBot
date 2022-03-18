'''
virtualenv venv
cd venv/Scripts
activate
cd ../..

python -m venv .venv
.\venv\Scripts\activate

pip install tensorflow
pip install tensorflow-gpu


pip install Flask
pip install nltk
pip install flask-ngrok
pip install flask==0.12.2

'''
'''
app.run(host="localhost", port=8000)

'''
from nltk import word_tokenize

'''
from nltk.tokenize import sent_tokenize
text_to_sentence = sent_tokenize(text)
print(text_to_sentence)
'''

'''
from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print(tokenized_word)
'''

#Filtering Stop Words:

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


#Removing Stop words:
'''
text = 'Learn to lose your destiny to find where it leads you'
filtered_text = []
tokenized_word = word_tokenize(text)
for each_word in tokenized_word:
    if each_word not in stop_words:
        filtered_text.append(each_word)

print(filtered_text)
'''
#Lemmatization:
'''
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

text = "Life will always have problems and pressures."
lemmatized_words_list = []
tokenized_word = word_tokenize(text)
for each_word in tokenized_word:
    lem_word = lemmatizer.lemmatize(each_word)
    lemmatized_words_list.append(lem_word)

print('Text with Stop Words: {}'.format(tokenized_word))
print('Lemmatized Words list {}'.format(lemmatized_words_list))
'''

#Parts of Speech(pos) tagging:
'''
import nltk
nltk.download('universal_tagset')

text = "I'm going to meet M.S. Dhoni."
tokenized_word = word_tokenize(text)
nltk.pos_tag(tokenized_word, tagset='universal')
text = "I'm going to meet M.S. Dhoni."
tokenized_word = word_tokenize(text)
nltk.pos_tag(tokenized_word, tagset='universal')
'''

# Named Entity Recognition:
'''
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')

text = "Sundar Pichai, the CEO of Google Inc. is walking in the streets of California."
tokenized_word = word_tokenize(text)
tags = nltk.pos_tag(tokenized_word, tagset='universal')
entities = nltk.chunk.ne_chunk(tags, binary=False)
print(entities)
'''

#WordNet:
'''
from nltk.corpus import wordnet
synonym = wordnet.synsets("AI")
print(synonym)
print(synonym[1].definition())
'''

# serialization
import pickle

#Here's an example dict
grades = { 'Alice': 89, 'Bob': 72, 'Charles': 87 }

#Use dumps to convert the object to a serialized string
serial_grades = pickle.dumps( grades )

#Use loads to de-serialize an object
received_grades = pickle.loads( serial_grades )