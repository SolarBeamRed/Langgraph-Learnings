import streamlit as st

from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# Display messages till now
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# Accepting user input
user_input = st.chat_input('Ask anything')

if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # LLM response
    config = {'configurable': {'thread_id': '1'}}
    response_state = chatbot.invoke({'messages': HumanMessage(content=user_input)}, config=config) # type: ignore
    ai_message = response_state['messages'][-1].content

    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)