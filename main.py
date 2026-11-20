from langchain_core.messages import(
    HumanMessage,AIMessage
)

from src.rag_pipeline import rag_chain

# 1. Conversation History Storage
print("Initiated.....")

store={}


# 2. Get Session History

def get_session_history(session_id):
    if session_id not in store:
        store[session_id]=[]
    return store[session_id] # Chat History of that particular session/user

# 3. Create user session
session_id="user_1"

chat_history=get_session_history(session_id)

# 4. start the Chatbot

print("\n Welcome to conversational RAG Chatbot")
print("\n Type exit to Stop the Chatbot")

# 5. Chat loop

while True:
    question=input("\nYou:")

    if question.lower()=="exit":
        print("Exiting from Chatbot...")
        break

    # 6. Intialte the RAG Chain
    answer=rag_chain.invoke({
        "input":question,
        "chat_history":chat_history[-10:]

    })

    print("\nAI:",answer)

    # 7. Store the HumanMessage and AIMessage in chat history

    chat_history.append(
        HumanMessage(content=question),
    )

    # 8. Store the AI message
    chat_history.append(
        AIMessage(content=answer)
    )
    