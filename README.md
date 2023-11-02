# Simple experiments using LangChain.AI

## Prerequisites

1. Install ollama using their [download instructions][1]. In another terminal
   run the llama2 model: `ollama run llama2`.
2. Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
3. Install Miniconda.
4. Create a conda env: `conda create -y -n langchain`
5. Activate conda env: `conda activate langchain`
6. Install FAISS: `conda install -c conda-forge faiss-cpu`
7. Install project dependencies: `poetry install`. (If you get a bunch of
   weird errors disable the python keyring and try again
   `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.)
8. Install pre-commit hooks: `pre-commit install`.

## twitter.py

Load my export of tweets and ask questions about them. 

`poetry run python langchain_experiments/twitter.py -l data/tweets.json <query>`
 
The `-l` flag is optional, if not provided it will try and read an existing
FAISS database. This takes a while, you have been warned. The output is pretty
garbage to be honest.

[1]: https://ollama.ai/download/
