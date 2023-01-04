#!/usr/bin/python3

# This part of the system is the actual chatbot, which follows these steps
#
#   1.  Load the preformatted podcast transcripts in memory from the CSV files 
#       generated with `embedder.py`
#
#   2.  Take a prompt as input
#
#   3.  Generates an embedding for the query string
#
#   4.  Sorts the in-memory database of podcast transcript embeddings for proximity with the input query embedding
#
#   5.  Enrich the OpenAI Completition prompt wich hallocination removal and the top-rated section from the sorted list at point 4.
#
#   6.  Send the prompt to OpenAI Completition API and shows the response
#
# The command line usage is the following
#
#   python3 all-in-chatbot.py question-string knowledge-dir

# CSV to load the podcast transcript kwowledge base
import csv
# Sys to handle command line arguments
import sys
# Pandas to import from CSV and handle dataframes
import pandas as pd
# Path to read all CSV files from a given dir
from pathlib import Path
# Os to read environment variable (API key)
import os
# OpenAI to get embedding of the query and query the Completitions API
import openai
# JSON to convert the string representing embedding (list of Float) into an actual list
import json
# Numpy to make the dot product between vector (embeddings)
import numpy as np

# Define function to calculate vector similarity
# As seen in https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

# Define the parameters for the Completion API.
# "Da Vinci" models have a 4000 chars prompt size limit
# For a list of available models, see https://beta.openai.com/docs/models/gpt-3
COMPLETIONS_API_PARAMETERS = {
    "temperature": 0.2,
    "max_tokens": 300,
    "model": "text-davinci-003"
}

# Check command line arguments
if( len(sys.argv) == 2 ):
   print ('Using transcript embedding in folder ', str(sys.argv[1]))
   embeddingsDir = sys.argv[1]
else:
   print ('Wrong number of arguments.')
   print ('Usage: ')
   print ('     python3 all-in-chatbot.py <transcripts-embedding-dir>')
   sys.exit()

# Read OpenAI API key
APIKEY = os.getenv('OPENAI_API_KEY')
if APIKEY is None:
   print ('There are no API keys set in the runtime environment.')
   print ('Set OPENAI_API_KEY in your environment before running this script.')
   sys.exit()
else:
   openai.api_key = APIKEY

# Read all the CVS files in memory
li = {}
files = Path(embeddingsDir).glob('*.csv')
for file in files:
        li[file.name] = pd.read_csv(file)
        li[file.name].insert( 2, column="episode", value=int(file.name[1:4]) )

allcsv = pd.concat(li, ignore_index=True)

# Prepare a array for the dot multiplication and related sorting
a = []
for i in range(len(allcsv)):
    a.append( (i, json.loads(allcsv['embedding'][i])) )

s = ""
while ( 1 ):

    # Ask user for the query input
    s = input( '\x1b[34m' + '\nWhat would you like to ask to All-in Pod Chatbot?     ' + '\x1b[0m')
    
    if( s == "exit" ):
        sys.exit("Good bye.")

    # If the user prompt is "tellmemore", the same prompt as before is used,
    # but the following relevant section are used from the sorted vector proximity
    # list. This is intended to provide the model with more information, which
    # could not be done otherwise given that prompt cannot exceed 4000 chars

    # TODO 001 This can be a good additional features

    # Get the embedding for the input query, using `text-embedding-ada-002` model for embedding generation
    # This should be the best model for the task as of December 2022
    query_embedding = openai.Embedding.create(model="text-embedding-ada-002", input=s)["data"][0]["embedding"]

    # Sort the list of content embeddings based on the vector similarity
    # with the input query string. The top result would be the section of content
    # most related to the query string, which will be then added to the Completion API 
    # prompt as context.
    res = sorted( [ (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in a ], reverse=True )

    # Completition API have a 4000 chars prompt size limit, so we need to add as much context as possible
    # making sure we still in this range.

    # Note: Removing the "say I don't know" part of the prompt make it more reliable
    # prompt_header = " Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say \"I don't know.\".\n\nContext:\n"
    prompt_header = " Answer the question as truthfully as possible using the provided context.\n\nContext:\n"

    prompt_tail = "\n\n Q: """ + s + """\n A: """

    prompt_context = ""

    prompt_size = len(prompt_header) + len(prompt_tail)

    n = 0
    while ( (prompt_size + len(allcsv['content'][res[n][1]])) < 4000 ):
        prompt_context += "\n* " + "[E " + str(allcsv['episode'][res[n][1]]) + " " + allcsv['time'][res[n][1]] + "] "
        prompt_context += allcsv['content'][res[n][1]]
        n += 1
        prompt_size += len( prompt_context )

    prompt = prompt_header + prompt_context + prompt_tail

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMETERS
    )

    # Debug message
    print ( '\x1b[92m' + "\nPrompt: " + prompt + '\x1b[0m', end='' )

    print ( '\x1b[95m' + response["choices"][0]["text"].strip(" \n") + '\x1b[0m' )
