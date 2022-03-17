# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import os
from os import listdir

lemmatizer = WordNetLemmatizer()


def compile_javascript():  # Defining the path to the folder where the JS files are saved
    path = 'static/javascript'  # Getting all the files from that folder
    files = [f for f in listdir(path) if isfile(join(path, f))]  # Setting an iterator
    i = 0  # Looping through the files in the first folder
    for file in files:  # Building a file name
        file_name = "javascript/" + file  # Creating a URL and saving it to a list
        all_js_files[i] = url_for('static', filename=file_name)  # Updating list index before moving on to next file
        i += 1
    return all_js_files


def compile_css():
    path = 'static/styles'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    i = 0
    for file in files:
        file_name = "styles/" + file
        all_js_files[i] = url_for('static', filename=file_name)
        i += 1
    return all_css_files


# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)


@app.route('/')
@app.route('/chatbot')
def home():
    all_js_files = compile_javascript()
    all_css_files = compile_css()
    return render_template('chatbot.html',
                           title='Home',
                           js_files=all_js_files,
                           css_files=all_css_files)


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    # checks is a user has given a name, in order to give a personalized feedback
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    # if no name is passed execute normally
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
