"""
Streamlit app using the rag_proof_of_concept package to implement a RAG chatbot
"""

import os, atexit
import rag_proof_of_concept as ragpoc
import streamlit as st

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

config = ragpoc.utils.load_config()

if config['chatbot']['verbose_logging']:
    logger.setLevel(logging.INFO)
    existing_loggers = list(logging.root.manager.loggerDict.keys())
    # Set logging level to INFO for all loggers under rag_proof_of_concept
    for logger_name in existing_loggers:
        if logger_name.startswith('rag_proof_of_concept'):
            logging.getLogger(logger_name).setLevel(logging.INFO)

if config['data']['http_server']:
    # Start a simple HTTP server to serve the data files
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    def start_server():
        httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
        httpd.serve_forever()

    # Start the server in a separate thread so that it doesn't block the main thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True  # So that the server dies when the main thread dies
    
    server_thread.start()


# Streamlit treats main() as state loop and re-runs it at every user input. The decorator of
# initialize_components ensures that the components are only initialized once.
@st.cache_resource
def initialize_components():
    client = ragpoc.vdb.get_weaviate_client(local=config['vdb']['local'])
    knowledge_vs = ragpoc.vdb.get_vector_store(client, config['vdb']['collection'])
    knowledge_connection = client.collections.get(config['vdb']['collection'])

    if config['vdb']['initialize_use_case_vs']:
        use_case_vs = ragpoc.vdb.initialize_use_case_vs(client)
    else:
        use_case_vs = ragpoc.vdb.get_vector_store(client, "RAG_use_cases")

    llm = ragpoc.utils.LLMClient(model=config['llm']['model'])

    # Register the close_client function to be called when the python interpreter shuts down
    atexit.register(ragpoc.vdb.close_client, client)

    return client, knowledge_vs, knowledge_connection, use_case_vs, llm

# This function is intended to be called when the user clicks the "Refresh your knowledge" button.
def refresh_vdb(client, llm):
    with st.spinner('Refreshing knowledge...'):
        try:
            ragpoc.readers.create_collection(
                client, 
                config['vdb']['collection'],
                config['data']['path'],
                config['vdb']['chunk_size'],
                config['vdb']['sentences_per_summary'],
                llm, 
                include_chunk_summaries = config['vdb']['include_chunk_summaries'],
                batch_size=config['vdb']['summarization_batch_size'])
            with open("conversation_context.log", "w") as f:
                f.write(st.session_state.conversation_context)
            logger.info("The vector store has been refreshed.")
        except Exception as e:
            logger.error("Failed to refresh the vector store: " + str(e))

# This function is intended to be called when the user clicks the "Delete your knowledge" button.
def clear_vdb(client):
    try:
        ragpoc.vdb.clear_vector_store(client, config['vdb']['collection'])
        logger.info("The vector store has been cleared.")
    except Exception as e:
        logger.error("Failed to clear the vector store: " + str(e))

# This function is intended to be called when the user clicks the "Forget our conversation" button.
def clear_conversation():
    st.session_state.conversation_context = ""
    with open("conversation_context.log", "w") as f:
            f.write(st.session_state.conversation_context)
    logger.info("The conversation context has been cleared.")

# The main function of the Streamlit app, which is re-run at every user input.
def main():
    logger.info("New chat iteration ----------------------------------------------------------------------------------------")
    client, knowledge_vs, knowledge_connection, use_case_vs, llm = initialize_components()
    try: 
        # The estimated chunk sizes is used to ensure that the target input token number is used in
        # LLM queries.
        expected_chunk_size = ragpoc.context.get_context_element_sizes(knowledge_connection)
    except Exception as e:
        logger.warning("Failed to estimate the chunk sizes: " + str(e))
    # Streamlit app layout
    st.title("Chat with your data.")

    # The message history is stored to render the full history of the session at each iteration
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # The conversation context is stored in the session_state and in a file to maintain the context
    # of the conversation across streamlit sessions.
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ""

        if os.path.exists("conversation_context.log"):
            with open("conversation_context.log", "r") as f:
                st.session_state.conversation_context = f.read()

    # Render the session message history
    for message in st.session_state.messages:
        # st.chat_message takes an argument `name`. The values "user" and "assistant" enable
        # preset avatars and styling.
        with st.chat_message(message["role"]):
            st.write(message["content"])
   
    # Buttons shown in the sidebar of the streamlit app
    st.sidebar.button(
        ":material/refresh: Refresh your knowledge",
        key = "refresh_button",
        use_container_width = True,
        on_click = lambda: refresh_vdb(client, llm))
    
    st.sidebar.button(
        ":material/delete: Delete your knowledge",
        key = "delete_button",
        use_container_width = True,
        on_click = lambda: clear_vdb(client))
    
    st.sidebar.button(
        ":material/clear_all: Forget our conversation",
        key = "clear_conversation_button",
        use_container_width = True,
        on_click = lambda: clear_conversation())

    # Process the input for this iteration
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            # We store the message container in a variable to display information about processing
            # steps while the agent is working
            msg_container = st.empty()

            raw_answer, formatted_answer = ragpoc.agent.core.get_answer(
                query_text=prompt,
                tokens_target=config['llm']['input_tokens_target'],
                expected_chunk_size=expected_chunk_size,
                knowledge_vs=knowledge_vs,
                knowledge_connection=knowledge_connection,
                use_case_vs=use_case_vs,
                history=st.session_state.conversation_context,
                style=config['chatbot']['conversation_style'],
                llm=llm,
                msg_container=msg_container
            )
            logger.info(raw_answer)
            msg_container.write(formatted_answer)
            
        st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

        st.session_state.conversation_context = ragpoc.agent.hist.update_conversation_context(
            st.session_state.conversation_context,
            prompt,
            ragpoc.agent.refs.strip_references(raw_answer), 
            llm)
        
        with open("conversation_context.log", "w") as f:
            f.write(st.session_state.conversation_context)

if __name__ == "__main__":
    main()