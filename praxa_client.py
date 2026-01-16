import streamlit as st
import praxa_rag

st.title("Praxa")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Ask me about the theater!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    response = praxa_rag.answer_and_sources(question)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        st.markdown(response["sources"])
    # Add assistant response to chat histore
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})