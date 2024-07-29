import torch
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM
def main():
    def load_dataset(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data

        def embed_data(data):
        # Tokenize data
            inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True)

        # Get word embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            return outputs.last_hidden_state

    # Load combined_dataset.txt
    file_path = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/combined_dataset.txt'  # Update this with the actual path if needed
    data = load_dataset(file_path)

    
    with open('C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/combined_dataset.txt', 'r',encoding='utf-8') as file:
        data = file.readlines()
    df = pd.DataFrame(data, columns=['text'])

    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

    train_df.to_csv('train_dataset.txt', index = False, header = False)
    test_df.to_csv('test_dataset.txt', index = False, header = False)
    print("data has been splitted successfully!")


if __name__ == "__main__":
    main()
    # print(data[:20])