import os
import pandas as pd
import numpy as np
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import streamlit as st

# Ensure the current working directory is the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Define the path to the CoQA dataset file
coqa_file_path = 'dev-v1.0 (1).json'

# Check if the file exists
if not os.path.exists(coqa_file_path):
    raise FileNotFoundError(f"File {coqa_file_path} does not exist. Please make sure the file is in the correct directory.")

# Load the CoQA dataset from the JSON file into a pandas DataFrame
coqa = pd.read_json(coqa_file_path)

# Delete the column named "version" from the coqa DataFrame
del coqa["version"]

# Specify the required columns for our new DataFrame
cols = ["text", "question", "answer"]

# Initialize an empty list to hold lists of data for creating the new DataFrame
comp_list = []

# Iterate over each row in the coqa DataFrame
for index, row in coqa.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])
        comp_list.append(temp_list)

# Create a new DataFrame from the list of lists, with the specified columns
new_df = pd.DataFrame(comp_list, columns=cols)

# Save the new DataFrame to a CSV file for further use
new_df.to_csv("CoQA_data.csv", index=False)

# Load the data from the CSV file 'CoQA_data.csv' into a pandas DataFrame
data = pd.read_csv("CoQA_data.csv")

# Print the number of question and answer pairs in the DataFrame
st.write("Number of question and answers: ", len(data))

# Load the pre-trained BERT model for question answering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def question_answer(question, text):
    input_ids = tokenizer.encode(question, text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)
    
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    else:
        answer = "Unable to find the answer to your question."
    
    return answer.capitalize()

# Streamlit app interface
st.title("Question Answering System")

context = st.text_area("Context")
question = st.text_input("Question")

if st.button("Get Answer"):
    answer = question_answer(question, context)
    st.write("Answer:", answer)
