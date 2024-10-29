# EBA DOCS

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
import os

DATA_PATH = os.getcwd() + r"\experiments\eba_docs"
db_collection = "VSNameCollection"

def main():

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    client = weaviate.connect_to_local()
    WeaviateVectorStore.from_documents(chunks, 
                                       embedding = OpenAIEmbeddings(), 
                                       client = client, 
                                       index_name = db_collection)
    client.close()

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

if __name__ == "__main__":
    main()