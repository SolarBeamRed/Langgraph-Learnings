from typing import Annotated, TypedDict

import sqlite3

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# LLM 
llm = ChatOpenAI(
    base_url='http://127.0.0.1:9931/v1',
    model='local qwen3.5:4b',
    api_key='not needed' # type: ignore
)


# State
class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes
def chat_node(state: ChatbotState):
    messages = state['messages']
    response = llm.invoke(messages)

    return {'messages': [response]}


#Sqlite database
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Graph
graph = StateGraph(ChatbotState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

checkpointer = SqliteSaver(connection)
chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) #type: ignore

    return list(all_threads)
