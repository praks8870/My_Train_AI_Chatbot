# MyTrain Chatbot

## Problem Statement:
The objective of the MyTrain Chatbot project is to develop an AI-powered chatbot that facilitates users in accessing Indian train time table information. The chatbot should prompt users for input regarding the "from" and "to" locations in multiple regional languages, as well as their seat/coach preferences (Optional), through text or audio inputs. Based on this input, the chatbot should retrieve and display all available trains on that route.

## Tools Used:
Python, PYtorch, PostgreSQL, Tkinter, NLTK, JSON, Speech Recognition.

## How to use the App:
Step-1:
Please Install all the required modules given.

Step-2:
Create a virtual environment in your IDE or in the Command prompt or in the Powershell using Anaconda.

Step-3:
Run the python file main_app.py in the created virtual environment.

Step-4:
The app will pop out in your screen.

Step-5: 
Use the Text or voice function to give inputs to get the response from the bot.

Step-6:
Use The short form of the junction or station names to get the details of the train like departure time, arrival time and more.

## Coding Steps:
Step-1:
First thing to do is to train our model using the intens stored in a json file. For training the model we need to get the data in numpy array format, since the ML or DL model will not recognize any data apart from integers or float values. For that we creating functions to Tokenize the given sentences the use stemming and put them in a bag of words function. This is the initial step.

Step-2:
Now we are summarise the model and include the layers for the model. For the model we are using Neural Networks from Pytorch Module. After importing the model we are tuning the layers hyper parameters and the activation functions.

Step-3:
in this step the data in the bag of words in passed through the layers of Neural network to train the model.

Step-4:
Then the chatbot app is created with the Tkinter module. 

Step-5:
By using the voice and text input to communicate with the chatbot and get the required responses from the chatbot.

