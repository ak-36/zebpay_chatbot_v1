import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.llms.openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
import csv
import openai

# Define Streamlit app layout and title
st.set_page_config(page_title="Cryptocurrency Chatbot", page_icon="💬", layout="centered")
st.title("Cryptocurrency Chatbot 💬")
openai.api_key = st.secrets.openai_key
portkey_key = st.secrets.portkey_key
# Define chat engine creation function
@st.cache_resource(show_spinner=False)
def CEngine():
    documents = SimpleDirectoryReader(input_files=["1.docx", "2.docx", "3.docx"]).load_data()
    headers = createHeaders(api_key=portkey_key, mode="openai")
    llm = OpenAI(api_base=PORTKEY_GATEWAY_URL, default_headers={
        "x-portkey-api-key": portkey_key,
        "x-portkey-provider": "openai",
        "Content-Type": "application/json"}, model="gpt-4"
    )
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "You are a customer support chatbot and an expert in cryptocurrency."
        ),
    )
    return chat_engine

@st.cache_resource(show_spinner=False)
def EscalationEngine():
    reader = SimpleDirectoryReader(
        input_files=["chat_history.docx"]
    )
    docs = reader.load_data()
    esc_llm = OpenAI(model="gpt-4")
    service_context = ServiceContext.from_defaults(llm=esc_llm, embed_model="local:BAAI/bge-small-en-v1.5")
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    memory = ChatMemoryBuffer.from_defaults(token_limit=15000)
    chat_engine = vector_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "Your role is to analyze the conversation, and see if the user needs an escalation to customer executive support or not. Return True if escalation needed else return False."
        ),
    )
    return chat_engine


chat_engine = CEngine()
esc_engine = EscalationEngine()

# Streamlit chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Enter your query:", key="user_input")

if st.button("Send"):
    if user_input.lower() == "exit":
        st.stop()
    response = chat_engine.stream_chat(user_input)
    esc_input = f"User Query: {user_input}, Bot Response: {response}"
    esc_response = esc_engine.chat(esc_input)
    if str(esc_response).lower() == "true":
        st.session_state.messages.append({"user": user_input, "bot": " ".join(token for token in response.response_gen) + "\nConnecting you with customer support"})
    else:
        st.session_state.messages.append({"user": user_input, "bot": " ".join(token for token in response.response_gen)})

for message in st.session_state.messages:
    st.write(f"User: {message['user']}")
    st.write(f"Bot: {message['bot']}")
