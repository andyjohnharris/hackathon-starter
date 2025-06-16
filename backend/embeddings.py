import openai
import os
import pandas as pd
import numpy as np
import json
import tiktoken
import psycopg2
import ast
import pgvector
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv, find_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Get openAI api key by reading local .env file
_ = load_dotenv(find_dotenv()) 
# openai.base_url = os.environ['OPENAI_API_BASE_URL']
# openai.api_key  = os.environ['OPENAI_API_KEY'] 

# Model and provider configuration
model = OpenAIModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url=os.environ['OPENAI_API_BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY'],
    ),
)
agent = Agent(model)

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('data/fam.csv')
df.head()

# Helper functions to help us create the embeddings

# Helper func: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Helper function: calculate length of essay
def get_essay_length(essay):
    word_list = essay.split()
    num_words = len(word_list)
    return num_words

# Helper function: calculate cost of embedding num_tokens
# Assumes we're using the text-embedding-ada-002 model
# See https://openai.com/pricing
def get_embedding_cost(num_tokens):
    return num_tokens/1000*0.0002

# Helper function: calculate total cost of embedding all content in the dataframe
def get_total_embeddings_cost():
    total_tokens = 0
    for i in range(len(df.index)):
        text = df['Name'][i]
        text = text + " " + df['DOB'][i]
        text = text + " " + df['Gender'][i]
        token_len = num_tokens_from_string(text)
        total_tokens = total_tokens + token_len
    total_cost = get_embedding_cost(total_tokens)
    return total_cost
  
# quick check on total token amount for price estimation
# total_cost = get_total_embeddings_cost()
# print("estimated price to embed this content = $" + str(total_cost))

# Create new list with small content chunks to not hit max token limits
# Note: the maximum number of tokens for a single request is 8191
# https://platform.openai.com/docs/guides/embeddings/embedding-models

# list for chunked content and embeddings
new_list = []
# Split up the text into token sizes of around 512 tokens
for i in range(len(df.index)):
    text = df['Name'][i]
    text = text + " " + df['DOB'][i]
    text = text + " " + df['Gender'][i]
    token_len = num_tokens_from_string(text)
    if token_len <= 512:
        new_list.append([df['Name'][i], df['DOB'][i], df['Gender'][i], token_len])
    else:
        # add content to the new list in chunks
        start = 0
        ideal_token_size = 512
        # 1 token ~ 3/4 of a word
        ideal_size = int(ideal_token_size // (4/3))
        end = ideal_size
        #split text by spaces into words
        words = text.split()

        #remove empty spaces
        words = [x for x in words if x != ' ']

        total_words = len(words)
        
        #calculate iterations
        chunks = total_words // ideal_size
        if total_words % ideal_size != 0:
            chunks += 1
        
        new_content = []
        for j in range(chunks):
            if end > total_words:
                end = total_words
            new_content = words[start:end]
            new_content_string = ' '.join(new_content)
            new_content_token_len = num_tokens_from_string(new_content_string)
            if new_content_token_len > 0:
                new_list.append([df['Name'][i], new_content_string, df['Gender'][i], new_content_token_len])
            start += ideal_size
            end += ideal_size

# openai_client = openai.OpenAI()

# Helper function: get embeddings for a text
def get_embeddings(text):
    response = model.embeddings.create(
        model="text-embedding-3-small",
        input = text.replace("\n"," ")
    )
    # response = openai_client.embeddings.create(
    #     model="text-embedding-3-small",
    #     input = text.replace("\n"," ")
    # )
    return response.data[0].embedding
  
# Create embeddings for each piece of content
for i in range(len(new_list)):
   text = new_list[i][1]
   embedding = get_embeddings(text)
   new_list[i].append(embedding)

# Create a new dataframe from the list
df_new = pd.DataFrame(new_list, columns=['title', 'content', 'url', 'tokens', 'embeddings'])
df_new.head()

print(df_new)
