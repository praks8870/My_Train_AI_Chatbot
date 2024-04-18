import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import psycopg2
from psycopg2 import sql
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr

stemmer = PorterStemmer()

def tokenize(sentence):

    return nltk.word_tokenize(sentence)


def stem(word):

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
with open('intents.json', 'r') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore = ['?', '!', '.', ',']
all_words = [stem(word) for word in all_words if word not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag= bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


#Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
# print(input_size, output_size)

class Chatbotdataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
dataset = Chatbotdataset()
train_loader = DataLoader(dataset= dataset,
                          batch_size= batch_size,
                          shuffle= True,
                          num_workers= 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # if (epoch+1) % 100 == 0:
        # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


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


bot_name = "Kash"
# print("Let's chat! (type 'quit' to exit)")



class ChatApplication(tk.Tk):
    def __init__(self, model, all_words, tags):
        super().__init__()
        self.model = model
        self.all_words = all_words
        self.tags = tags
        self.title("Chatbot App")

        self.chat_history = scrolledtext.ScrolledText(self, width=40, height=10)
        self.chat_history.pack(pady=10)

        self.user_input = tk.Entry(self, width=40)
        self.user_input.pack(pady=5)

        self.voice_button = tk.Button(self, text="Voice Input", command=self.get_voice_input)
        self.voice_button.pack(pady=5)

        self.user_input.bind("<Return>", self.send_message)

    def get_voice_input(self):
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            user_text = recognizer.recognize_google(audio)
            print("You said:", user_text)
            self.user_input.delete(0, tk.END)
            self.user_input.insert(tk.END, user_text)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

    def send_message(self, event=None):
        user_text = self.user_input.get()
        self.user_input.delete(0, tk.END)
        self.chat_history.insert(tk.END, "You: " + user_text + "\n")
        response = self.get_response(user_text)
        self.chat_history.insert(tk.END, "Kash: " + response + "\n")
        
    def get_response(self, user_text):
        sentence = tokenize(user_text)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        tag_list = ['greeting', 'goodbye', 'thanks', 'items', 'payments', 'funny', 'trains']

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    if tag == 'search_train':
                        # Get the train name from the user input
                        inp = user_text
                        # Check train details using the provided train name
                        return sql_find(inp)
                    elif tag in tag_list:
                        return random.choice(intent['responses'])
                    break 
        else:
            return "I do not understand..."

# Initialize and run the Tkinter application
app = ChatApplication(model, all_words, tags)
app.mainloop()

