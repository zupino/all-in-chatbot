# All-in Podcast Chatbot

This script makes use of ChatGPT to answer user questions based on your favorite podcast content, a useful tool to quickly access the content of podcast episodes and provide time reference where the topic requested was discussed.

OpenAI Completion API are used to handle the language processing, while Embedding API are used to find relevant sections of podcast transcript to use as context, which is then added to the textual prompt. 

This work was inspired by Superorganizer blog post [1] and was realized as a personal project to search within the content of the All-in Podcast [2] episodes, providing answer based on the episode content and reference to the episode in which the topic was discussed. 

# System components

The system operates with two components, `embedder.py` and `all-in-chatbot.py`.

`embedder.py` takes the transcripts of the podcasts episodes as input and generate a csv files contains sections of the transcripts with the related OpenAI Embeddings [3] for each section.

`all-in-chatbot.py` loads in memory the pre-processed dataset and uses it to find sections of the transcript which are relevant to the question provided by the user, and add them as context to the OpenAI Completion API prompt. Vector distance is used to find the sections which are more relevant to the input question.

# Notes and improvements

The quality of the transcript affects the system answers, especially for names. 

Lenght of section is considered when filtering to generate the embeddings, which might be affected on the text format of the podcast transcript. Transcript text should be further normalized before processing.

The system works with a simple command line text interface, which I like for my personal use. Some might prefer a nicer UI or API endpoints.

There is a limit on the size of the prompt that can be used with the OpenAI Completion API, which depends on the model used. This factor limits the amount of sections which can be added as prompt context.

# Conclusion

The most interesting part for me was to use OpenAI Embedding to place user questions and sections of podcast transcript in a vector space where relevance can then be determined as distance between vectors. I always find this idea fascinating and would love to look deeper into the topic. 

[1] https://every.to/superorganizers/i-trained-a-gpt-3-chatbot-on-every-episode-of-my-favorite-podcast
[2] https://www.allinpodcast.co
[3] https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
