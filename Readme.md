# MyTrain Chatbot

## Problem Statement:
The objective of the MyTrain Chatbot project is to develop an AI-powered chatbot that facilitates users in accessing Indian train time table information. The chatbot should prompt users for input regarding the "from" and "to" locations in multiple regional languages, as well as their seat/coach preferences (Optional), through text or audio inputs. Based on this input, the chatbot should retrieve and display all available trains on that route.

## Overview
This Chatbot Application is built using Flask and PyTorch. It utilizes natural language processing (NLP) techniques to understand user queries related to train information. The chatbot can respond to greetings, farewells, expressions of gratitude, jokes, and queries about trains.

## Features
- **Conversational Interface**: The chatbot provides a seamless conversation experience where users can chat and receive instant responses.
- **Voice Input**: Users can interact with the chatbot using voice commands.
- **Train Information Retrieval**: The chatbot fetches train details from a PostgreSQL database based on user queries.

## Technologies Used
- **Flask**: Web framework for creating the application.
- **PyTorch**: Deep learning library for training the chatbot model.
- **NLTK**: Natural Language Toolkit for text processing.
- **NumPy**: For numerical computations.
- **PostgreSQL**: Database to store and retrieve train information.
- **SpeechRecognition**: For processing voice inputs.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/praks8870/My_Train_AI_Chatbot.git

2. **Navigate to the project directory**:
    ```bash
    cd My_Train_AI_Chatbot

3. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. **Install required packages:**
    ```bash
    pip install -r requirements.txt

5. **Set up the PostgreSQL database**: 

    Ensure you have PostgreSQL installed and running. Create a database and the required tables as per your application needs.

6. **Run the application**:
    ```bash
    python app.py

7. **Access the chatbot**: 

Open your web browser and navigate to http://127.0.0.1:5000.

## Usage:
- Start chatting with the bot by typing your message in the input field and pressing "Send" or hitting Enter.
- Use the "Voice Input" button to speak to the bot.

## Contributing:
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.

## Acknowledgments:
- Thanks to the contributors of the libraries used in this project.
- Special thanks to the open-source community for their continuous support and contributions.