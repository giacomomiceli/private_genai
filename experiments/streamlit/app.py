import argparse
import weaviate
import os
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import streamlit as st


# Chatbot model
llm=ChatOpenAI(model="gpt-4o")

# Collection of all docs
db_collection = "VSNameCollection"

def get_openai_response(query_text):

    # Connect to a WCS instance
    # client = weaviate.connect_to_wcs(
    #     cluster_url=os.getenv('WEAVIATE_URL'),
    #     auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY')))
    client = weaviate.connect_to_local()


    # DB connection
    vs=WeaviateVectorStore(client = client,
                           index_name = db_collection,
                           embedding = OpenAIEmbeddings(),
                           text_key = "text")
    
    template = """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Use five sentences minimum and keep the answer concise.
                Question: {question} 
                Context: {context} 
                Answer:
                Source:
                Page:
                """
    
    retriever=vs.as_retriever()

    ## Retrieve context and Generate answer
    prompt = ChatPromptTemplate.from_template(template)
    # Retriever, prompt, and language model workflow
    rag_chain=(
        {"context": retriever,"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response=rag_chain.invoke(query_text)
        client.close()
        return response if response else "No response from the model."
    
    except Exception as e:
        client.close()
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":

    # query_text = "Tell me about EBA"
    # response = get_openai_response(query_text)
    # print(response)

    # Streamlit app layout
    st.title("Chat with your data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            stream = get_openai_response(prompt)
            #response = st.write_stream(stream)
            response = st.write(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # user_input = st.text_input("What would you like to know from your data?")
    # if st.button("Submit"):
    #     chatbot_response = get_openai_response(user_input) if user_input else "Please enter a question or message to get a response."
    #     st.write(f"Chatbot: {chatbot_response}")