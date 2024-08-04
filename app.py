# Environment Dependencies
import os
import importlib

# Webapp Dependencies
import gradio as gr
import pandas as pd

# ML Dependencies
import torch
import tensorflow as tf
from transformers import pipeline

# LLM Dependencies
import cohere as co
import openai as oa
import huggingface_hub as hf

#Load API Keys
openai_key = os.getenv('OPENAI_API_KEY')
cohere_key = os.getenv('COHERE_API_KEY')
huggingface_key = os.getenv('HUGGINGFACE_API_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')
claude_key = os.getenv('CLAUDE_API_KEY')

# Verify Installation
def verify_installation():
    libraries = ['torch', 'tensorflow', 'transformers']
    for lib in libraries:
        try:
            lib_obj = importlib.import_module(lib)
            print(f"{lib.capitalize()} is installed successfully!")
        except ImportError:
            print(f"Error: {lib.capitalize()} is not installed.")

    # Additional verification for PyTorch and TensorFlow by creating a random tensor
    try:
        torch.manual_seed(42)
        x = torch.rand(5, 3)
        if x is not None:
            print("PyTorch: Installation verified successfully!")
    except NameError:
        print("PyTorch: Unable to verify installation (torch.rand not found)")

    try:
        
        tf.random.set_seed(42)
        x = tf.random.normal([5, 3])
        if x is not None:
            print("TensorFlow: Installation verified successfully!")
    except NameError:
        print("TensorFlow: Unable to verify installation (tf.random.normal not found)")

verify_installation()


# Load Models and APIs

# Gradio based User Interface
def 