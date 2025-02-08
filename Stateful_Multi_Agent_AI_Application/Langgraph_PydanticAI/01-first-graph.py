import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-06-01-preview"


class State(TypedDict):
    messages: Annotated[list, add_messages]
    

def first_node(state: State):
    
    llm = AzureChatOpenAI(
        openai_api_version="2023-06-01-preview",
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        streaming=True
    )
    
    system_message = SystemMessage("You are a helpful AI assistant.")
    
    messages = [system_message] + state['messages']
    
    return {"messages": [llm.invoke(messages)]}


def create_graph():
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("first_node", first_node)
    
    graph_builder.add_edge(START, "first_node")
    graph_builder.add_edge("first_node", END)
    
    return graph_builder.compile()


def main():
    graph = create_graph()
    
    query = input("Your query: ")
    
    initial_state = {"messages": [HumanMessage(content=query)]}
    
    for event in graph.stream(initial_state):
        for key in event:
            print("\n**************************\n")
            print(key + ":")
            print("------------------------------\n")
            print(event[key]['messages'][-1].content)
            
            
if __name__ == "__main__":
    main()