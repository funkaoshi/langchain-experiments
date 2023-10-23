# Simple experiments using LangChain.AI

## Prerequisites

1. Install ollama using their [download instructions][1]. In another terminal
   run the llama2 model: `ollama run llama2`.
2. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install project dependencies: `poetry install`. (If you get a bunch of
   weird errors disable the python keyring and try again
   `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.)
4. Install pre-commit hooks: `pre-commit install`.

## twitter.py

Load my export of tweets and ask questions about them. 

`poetry run python langchain_experiments/twitter.py -l data/tweets.json <query>`
 
The `-l` flag is optional, if not provided it will try and read an existing
chromadb database. On my Mac loading all of my tweets into Chroma is so slow
I haven't managed to load them all. I have been testing loading 100 tweets.

[1]: https://ollama.ai/download/
