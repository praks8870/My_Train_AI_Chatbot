from flask import Flask, render_template, request, session
import torch
import json
import random
import psycopg2
from psycopg2 import sql
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Load your trained model and required data
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

# Load the model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as f:
    intents = json.load(f)

data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load model
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size) 
        self.l2 = torch.nn.Linear(hidden_size, hidden_size) 
        self.l3 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(user_text):
    sentence = tokenize(user_text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    tag_list = ['greeting', 'goodbye', 'thanks', 'items', 'payments', 'funny', 'trains']

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                if tag == 'search_train':
                    return sql_find(user_text)
                elif tag in tag_list:
                    return random.choice(intent['responses'])
                break
    else:
        return "I do not understand..."

def sql_find(train_name):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="123456",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    insert_query = sql.SQL("""
        SELECT "Train No", "Train Name", "Station Name", "Destination Station Name",
               "Arrival Time", "Departure Time", "Distance"
        FROM train_schedule
        WHERE "Search" LIKE %s
        LIMIT 1 """)

    cur.execute(insert_query, (f'%{train_name}%',))
    result = cur.fetchone()

    cur.close()
    conn.close()

    if result:
        train_no, train_name, station_name, dest_station_name, arrival_time, departure_time, distance = result
        return f"Train No: {train_no}, Train Name: {train_name}, Station: {station_name}, Destination Station: {dest_station_name}, Arrival Time: {arrival_time}, Departure Time: {departure_time}, Distance: {distance}"
    else:
        return "No train found."

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = get_response(user_message)

    # Update chat history
    session['chat_history'].append({'user': user_message, 'bot': response})

    # Save the session
    session.modified = True

    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == "__main__":
    app.run(debug=True)
