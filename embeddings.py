import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from mistralai.client import MistralClient
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from datasets import concatenate_datasets, load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# def load_data(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = file.read()
#     return data

# def get_chunks(data):
#     text_splitter = CharacterTextSplitter(
#         chunk_size = 100000,
#         separator = "\n",
#         chunk_overlap = 200,
#         length_function = len
#     )
#     chunks = text_splitter.split_text(data)
#     return chunks

# def get_vectorstore(chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    
#     # Create vectorstore from embeddings
#     vectorstore = FAISS.from_texts(texts=chunks, embedding=lambda x: embeddings[chunks.index(x)])
#     return vectorstore


# def main():
#     load_dotenv()
#     data = load_data('C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/combined_dataset.txt')
#     chunks = get_chunks(data)
#     vectorstores = get_vectorstore(chunks)
#     # for i, chunk in enumerate(chunks):
#     #     print(f"chunk {i}: {chunk[:100]}...")

# if __name__ == "__main__":
#     main()




import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Define the paths to the dataset files
train_dataset_path = "C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/train_dataset.txt"
test_dataset_path = "C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/test_dataset.txt"
base_model_id = "hkunlp/instructor-xl"  # Replace with your actual model ID
max_length = 1024  # Max length for chunking embeddings

# Load the model and tokenizer
def load_model_and_tokenizer(model_id):
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Function to read datasets from file paths
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

# Function to generate embeddings
def generate_embeddings(model, tokenizer, dataset):
    embeddings = []
    for text in dataset:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate embeddings
        with torch.no_grad():
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Assuming you want embeddings from the last hidden state
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state.mean(dim=1).squeeze().tolist())
    
    return embeddings

# Function to chunk embeddings
def chunk_embeddings(embeddings, chunk_size):
    num_chunks = (embeddings.shape[0] + chunk_size - 1) // chunk_size
    chunks = np.array_split(embeddings, num_chunks)
    return chunks

# Function to plot the distribution of embedding lengths
def plot_embedding_lengths(embeddings):
    lengths = [embedding.shape[0] for embedding in embeddings]
    
    print(f"Total sequences: {len(lengths)}")

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of Embeddings')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of Embeddings')
    plt.show()

def main():
    model, tokenizer = load_model_and_tokenizer(base_model_id)
    
    train_dataset = read_dataset(train_dataset_path)
    test_dataset = read_dataset(test_dataset_path)
    
    # Generate embeddings
    train_embeddings = generate_embeddings(model, tokenizer, train_dataset)
    test_embeddings = generate_embeddings(model, tokenizer, test_dataset)
    
    # Plot lengths before chunking
    plot_embedding_lengths([train_embeddings, test_embeddings])

    # Chunk the embeddings
    train_chunks = chunk_embeddings(train_embeddings, max_length)
    test_chunks = chunk_embeddings(test_embeddings, max_length)

    # Plot lengths after chunking
    plot_embedding_lengths(train_chunks)
    plot_embedding_lengths(test_chunks)

if __name__ == "__main__":
    main()
