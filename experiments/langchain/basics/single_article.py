import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore

llm=ChatOpenAI(model="gpt-4o")

def load_content_onto_vd(file_path):

    loader=TextLoader(file_path)
    docs=loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
    document=text_splitter.split_documents(docs)

    # Connect to a WCS instance
    # client = weaviate.connect_to_wcs(
    #     cluster_url=os.getenv('WEAVIATE_URL'),
    #     auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY')))

    client = weaviate.connect_to_local()

    vs=WeaviateVectorStore.from_documents(
        document,
        embedding=OpenAIEmbeddings(),
        client=client,
        index_name = "RAGchat"
    )

    client.close()    

    return()

# 1. Weaviate Client: Create a New Vector Database
## Instantiate a Weaviate client to create a new vector database. Run this process only if the database has not been created yet.
article_path = os.getcwd() + r"\experiments\articles\nyt_28Jul2024.txt"

load_content_onto_vd(file_path = article_path)

# 2. Instantiating Weaviate and deploying a vector database for enhanced answer retrieval and generation using LLM

# Connect to a WCS instance
# client = weaviate.connect_to_wcs(
#     cluster_url=os.getenv('WEAVIATE_URL'),
#     auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY'))) 

client = weaviate.connect_to_local()

# The NYT_28Jul2024.txt: "Fears of Escalation After Rocket From Lebanon Hits Soccer Field"
vs=WeaviateVectorStore(client=client, index_name="RAGchat", embedding=OpenAIEmbeddings(), text_key="text")

retriever=vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.6}
)

template= """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

## Retrieve context and Generate answer
prompt = ChatPromptTemplate.from_template(template)
# Retriever, prompt, and language model workflow
rag_chain=(
    {"context": retriever,"question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "Do you know Euan Ward?"

retriever.invoke("Who is Peppa Pig?")

response=rag_chain.invoke(input="Who is Peppa Pig?")

# response=rag_chain.invoke("Can you provide a summary or review of the most recent article by Patrick Kingsley, Euan Ward, and Isabel Kershner?")
print(response)

client.close()