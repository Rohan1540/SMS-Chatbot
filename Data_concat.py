import os
from datasets import load_dataset


def concat_text_files(file1, file2, output_file):
    # Read content of the first file
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = f1.read()

    # Read content of the second file
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = f2.read()

    # Concatenate the contents of the two files
    concatenated_data = data1 + "\n" + data2

    # Write the concatenated data to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(concatenated_data)

    print(f"Data from {file1} and {file2} has been concatenated into {output_file}")


def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return text_data



def combine_datasets(local_text_data, hf_dataset):
    combined_text = local_text_data
    for row in hf_dataset:
        combined_text += "\n" + row['input']  # assuming the column name is 'text'
    return combined_text



if __name__ == "__main__":
    # File paths
    local_text_file_path = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/Web_scrapped.txt'
    hf_dataset_url = 'hf://datasets/lavita/ChatDoctor-HealthCareMagic-100k/data/train-00000-of-00001-5e7cb295b9cff0bf.parquet'
    file1 = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/scraped_articles_extended.txt'
    file2 = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/neurological_conditions_articles.txt'
    output_file = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/Web_scrapped.txt'


    local_text_data = load_text_file(local_text_file_path)

    # Load Hugging Face dataset
    hf_dataset = load_dataset('parquet', data_files={'train': hf_dataset_url}, split='train')

    # Combine datasets
    combined_text = combine_datasets(local_text_data, hf_dataset)

    # Save the combined dataset to a new text file
    combined_text_file_path = 'C:/Users/KIIT/Desktop/SMS-ChatBot/Dataset/combined_dataset.txt'
    with open(combined_text_file_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"Datasets combined and saved to '{combined_text_file_path}'.")
    # Concatenate the files
    concat_text_files(file1, file2, output_file)
