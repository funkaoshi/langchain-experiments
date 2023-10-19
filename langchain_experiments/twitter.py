import argparse

from langchain.chains import RetrievalQA
from langchain.document_loaders import JSONLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["favorites"] = record.get("favorite_count")
    metadata["retweets"] = record.get("retweet_count")
    metadata["created_at"] = record.get("created_at")

    return metadata


def load_tweets_into_db(path: str):
    # Load all my tweets from a twitter export into langchain documents
    loader = JSONLoader(
        file_path=path,
        jq_schema=".[].tweet",
        content_key="full_text",
        metadata_func=metadata_func,
    )
    data = loader.load()

    logger.debug(f"{len(data)} LangChain documents created from tweets.")

    # break up the text of the tweets into smaller chunks stored in a vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    logger.debug(f"LangChain documents split into {len(splits)} chunks.")

    # todo: this is hella slow, hence the 100 tweet limit
    vectorstore = Chroma.from_documents(
        documents=splits[:100], embedding=OllamaEmbeddings(), persist_directory="./chroma_db"
    )

    logger.debug("VectorStore populated and ready to be queried.")

    return vectorstore


def load_tweets_from_db():
    vectorstore = Chroma(
        persist_directory="./chroma_db", embedding_function=OllamaEmbeddings()
    )

    logger.debug("VectorStore loaded from disk.")

    return vectorstore


def main(query: str, path: str | None):
    # Load tweets from a twitter export into a vector database
    if path:
        vectorstore = load_tweets_into_db(args.load_tweets)
    else:
        vectorstore = load_tweets_from_db()

    logger.debug("Connect to llama2 model running in Ollama")

    ollama = Ollama(base_url="http://localhost:11434", model="llama2")

    logger.debug("Create RetrievalQA chain we will query against.")

    qa = RetrievalQA.from_chain_type(
        llm=ollama,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        verbose=True,
    )

    prompt_template = PromptTemplate.from_template(
        """
        You are an AI bot that has read all of my tweets on twitter, they
        are the additional context you've been given to understand my queries.

        {query}
        """
    )

    logger.debug("Query the RetrievalQA chain.")

    logger.info(qa.run(prompt_template.format(query=query)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline twitter query.")
    parser.add_argument("query")
    parser.add_argument("-l", "--load-tweets", action="store")
    args = parser.parse_args()

    main(args.query, args.load_tweets)
