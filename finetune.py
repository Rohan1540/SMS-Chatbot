import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from embeddings import get_chunks, get_vectorstore, load_data

def prepare_training_data(chunks):
    # Assuming chunks are the text data prepared for training
    # Convert chunks to the format required for fine-tuning GPT-2
    return chunks

def fine_tune_gpt2(training_data, model_name='gpt2', output_dir='./fine_tuned_gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize the training data
    tokenized_data = tokenizer(training_data, return_tensors='pt', padding=True, truncation=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    data = load_data('C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/combined_dataset.txt')
    chunks = get_chunks(data)
    vectorstore = get_vectorstore(chunks)

    # Prepare the training data from the vectorstore (if needed)
    training_data = prepare_training_data(chunks)
    
    # Fine-tune the GPT-2 model
    fine_tune_gpt2(training_data)

if __name__ == "__main__":
    main()
