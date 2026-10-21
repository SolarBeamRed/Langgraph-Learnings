import streamlit as st
import uuid

from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot, llm



# ________________________  Utility  ______________________

def generate_uuid():
    return str(uuid.uuid4())


# When clicking new chat
def reset_chat():
    thread_id = generate_uuid()                     # Generate new thread_id
    st.session_state['thread_id'] = thread_id       # Use new thread_id for new convo
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []        # Clear previous messages


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = 'New Chat'


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    )

    return state.values.get('messages', [])
    

# _____________________ Session Startup ___________________

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Chat threads will be dict of key=thread_id and value=conversation_description
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = {}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_uuid()

add_thread(st.session_state['thread_id'])



# _____________________ Sidebar _______________________

st.sidebar.title('SkrrtPopPop Chatbot')
if st.sidebar.button('New Chat'):
    reset_chat()


st.sidebar.header('Your conversations')
for thread_id, title in reversed(list(st.session_state['chat_threads'].items())):
    if st.sidebar.button(title, key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        msg_list = []
    
        for msg in messages:
            role = 'human' if isinstance(msg, HumanMessage) else 'assistant'
            msg_list.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = msg_list


# ________________________ Main UI _______________________

# Display messages till now
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# Accepting user input
user_input = st.chat_input('Ask anything')

if user_input:
    thread_id = st.session_state['thread_id']
    if st.session_state['chat_threads'][thread_id] == 'New Chat':
        summary = llm.invoke(f"Explain the intent of this text in less than 8 words so that it can be used to display title of a conversation in a chatbot application: {user_input}").content
        st.session_state['chat_threads'][thread_id] = summary

    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # LLM response
    config = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream( #type: ignore
                config=config,   #type: ignore
                input={'messages': HumanMessage(content=user_input)},   #type: ignore
                stream_mode='messages'
            )
        )
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})