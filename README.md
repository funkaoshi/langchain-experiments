# Simple experiments using LangChain.AI

## Prerequisites


1. Install ollama using their [download instructions][1]. 
2. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install project dependencies: `poetry install`

## twitter.py

Load my export of tweets and ask questions about them. 

`poetry run python langchain_experiments/twitter.py -l data/tweets.json <query>`
 
The `-l` flag is optional, if not provided it will try and read an existing
chromadb database. On my Mac loading all of my tweets into Chroma is so slow
I haven't managed to load them all. I have been testing loading 100 tweets.

[1]: https://ollama.ai/download/
