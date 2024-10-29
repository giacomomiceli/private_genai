import os
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Define a function to transform special characters
def transform_text(text):
    #return text.replace('“', '”', '’').replace('"', '"', "'")
    text = (text.replace('“', '"')
            .replace('”', '"')
            .replace("’", "'"))
    return text

#------------------------------------------------------------------------------
# 1. Loading Multiple Files
#------------------------------------------------------------------------------

# Specify the directory containing your article files
articles_dir = r'c:/Users/...'

docs = []

# Iterate through each .txt file in the directory
for filename in os.listdir(articles_dir):
    if filename.endswith('.txt'):
        # Load the content of the file
        file_path = os.path.join(articles_dir, filename)
       
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # Metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Assuming the first line is the title and the rest is the content
        lines = content.splitlines()
        title = transform_text(lines[0]) if lines else "Untitled"  # Transform title
        article_content = transform_text("\n".join(lines[1:])) if len(lines) > 1 else ""

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Create Document objects with metadata
        for i, text in enumerate(texts):
            metadata = {
                'file_name': filename,
                'title': title,  # Assuming title is in metadata
                'chunk_id': i
            }
            doc = Document(page_content=text.page_content, metadata=metadata)
            docs.append(doc)

#------------------------------------------------------------------------------
# 2. Store the documents in Weaviate
#------------------------------------------------------------------------------

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv('WEAVIATE_URL'),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY')))
vs=WeaviateVectorStore.from_documents(docs,embedding=OpenAIEmbeddings(),client=client)

