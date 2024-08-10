# Environment Dependencies
import os
import importlib
import time
import random
import getpass

# Webapp Dependencies
import gradio as gr
import pandas as pd

# ML Dependencies
import torch
import tensorflow as tf
from transformers import pipeline

# LLM Dependencies
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI

from langchain_core.messages import AIMessage



#Load API Keys
openai_key = os.getenv('OPENAI_API_KEY')
cohere_key = os.getenv('COHERE_API_KEY')
huggingface_key = os.getenv('HUGGINGFACE_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')
claude_key = os.getenv('CLAUDE_API_KEY')

# Verify Installation
def verify_installation():
    libraries = ['torch', 'tensorflow']
    installed_bool = {"torch": False, "tensorflow": False}
    verified_bool = {"torch": False, "tensorflow": False}
    for lib in libraries:
        try:
            lib_obj = importlib.import_module(lib)
            #print(f"{lib.capitalize()} is installed successfully!")
            installed_bool[lib] = True
        except ImportError:
            #print(f"Error: {lib.capitalize()} is not installed.")
            pass

    # Additional verification for PyTorch and TensorFlow by creating a random tensor
    try:
        torch.manual_seed(42)
        x = torch.rand(5, 3)
        if x is not None:
            #print("PyTorch: Installation verified successfully!")
            verified_bool["torch"] = True
    except NameError:
       #print("PyTorch: Unable to verify installation (torch.rand not found)")
       pass

    try:
        
        tf.random.set_seed(42)
        x = tf.random.normal([5, 3])
        if x is not None:
            #print("TensorFlow: Installation verified successfully!")
            verified_bool["tensorflow"] = True
    except NameError:
        #print("TensorFlow: Unable to verify installation (tf.random.normal not found)")
        pass
verify_installation()


# Load Models and APIs

# Essential Chatbot Functions

def user(user_message, history):
    return "", history + [[user_message, None]]

def add_message(history, message):
    for x in message["files"]:
        history.append(((x["path"],), None))  
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False, file_types=["image"])

def bot(history):
        bot_message = random.choice(["My brother from a biological mother, how can i help you?", "How's it hanging?", "What's up, my G?"])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def greet(history, input):
    return history + [(input, "Hello, " + input)]

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])



# Gradio based User Interface
with gr.Blocks() as demo:
    with gr.Row(elem_id="root-container"):
        with gr.Column(elem_id="chat-history", scale = 1, variant = "compact"):
            chat_history = gr.List(label="Chat History", interactive=False, wrap = True, row_count=10, col_count= 1, headers= [""])
        with gr.Column(elem_id="chatbot", scale = 5, variant = "panel"):
            chatbot = gr.Chatbot()
            with gr.Row(elem_id="basic-chatbot-buttons"):
                undo_btn = gr.Button(value = "Undo", icon="assets/icons/undo.svg", variant="secondary")
                clear_btn = gr.Button(value = "Clear", icon="assets/icons/clear.svg", variant="secondary")
                new_conversation_btn = gr.Button(value = "New", icon="assets/icons/new_conversation.svg", variant="secondary")
            export_conversation_btn = gr.Button(value = "Export Conversation", icon="assets/icons/export.svg", variant="secondary")
            with gr.Accordion(elem_id="basic-chatbot-config", label="Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        select_language = gr.Dropdown(["English", "French", "Punjabi"], label="Language", value="English", interactive=True, multiselect= False)
                        select_save_format = gr.Dropdown(["Rich Text", "JSON", "CSV"], label="Save Format", value="Rich Text", interactive=True, multiselect= False)
                    with gr.Column():
                        select_return_key = gr.Radio(["Enter", "Shift+Enter"], label="Submit on", value="Enter", interactive=True)
            chat_input = gr.MultimodalTextbox(autofocus=True, autoscroll=True, placeholder="Type your message here or drag and drop files...", submit_btn=True, interactive=True, lines= 5, max_lines=20)

            
            # Chatbot Functions
            chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
            bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
            bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
            chatbot.like(print_like_dislike, None, None)
            clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()