from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st



# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant. Please answer the following question.\n"),
        ("user", "Question: {question}")
    ]
)

# Streamlit Framework - Bard-like UI
st.set_page_config(
    page_title="Intelligent Chat Assistant",
    page_icon="✨",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
        }
        .chat-header {
            text-align: center;
            font-size: 1.8em;
            font-weight: bold;
            color: #3b5998;
            margin-bottom: 20px;
        }
        .user-input {
            width: 100%;
            padding: 10px;
            border-radius: 25px;
            border: 1px solid #ddd;
            font-size: 1em;
        }
        .chat-bubble-user {
            text-align: right;
            margin: 10px 0;
        }
        .chat-bubble-user p {
            display: inline-block;
            background-color: #e9f5ff;
            padding: 10px 15px;
            border-radius: 15px;
            color: #333;
        }
        .chat-bubble-bot {
            text-align: left;
            margin: 10px 0;
        }
        .chat-bubble-bot p {
            display: inline-block;
            background-color: #f1f0f0;
            padding: 10px 15px;
            border-radius: 15px;
            color: #333;
        }
        .submit-button {
            margin-top: 10px;
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Main chat container
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

st.markdown("<div class='chat-header'>Smart Assistant 🤖</div>", unsafe_allow_html=True)

# Input box styled as a search bar
input_text = st.text_input("", placeholder="Ask me anything...", key="user_input")

# OpenAI LLM setup
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 500,
        "repetition_penalty": 1.2,
        "stop_sequence": ["\n"]
    },
    huggingfacehub_api_token="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"
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Display chatbot interaction
if st.button("Submit", use_container_width=True):
    if input_text:
        with st.spinner("🤔 Generating response..."):
            response = chain.invoke({'question': input_text})

        # User query bubble
        st.markdown(f"""
            <div class="chat-bubble-user">
                <p>{input_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # Bot response bubble
        st.markdown(f"""
            <div class="chat-bubble-bot">
                <p>{response}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")

st.markdown("</div>", unsafe_allow_html=True)  # Close chat container
