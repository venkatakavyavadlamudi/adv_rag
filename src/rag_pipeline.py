
from src.config import retriever

from src.chains import (
    intent_chain,
    contextualize_chain,
    document_answer_chain,
    history_answer_chain,
    general_chain
)

from langchain_core.runnables import RunnableLambda


def retrieve_documents(inputs):

    # Rewrite question using conversation history
    standalone_question = contextualize_chain.invoke({
        "input": inputs["input"],
        "chat_history": inputs["chat_history"]
    })

    print("\nRewritten Question:", standalone_question)

    # Retrieve relevant documents from Supabase
    documents = retriever.invoke(standalone_question)

    # Combine retrieved document content
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    if not context.strip():
        return "I Don't Know based on the provided document."

    # Generate answer using PDF context only
    answer = document_answer_chain.invoke({
        "input": standalone_question,
        "context": context
    })

    return answer


def route_request(inputs):

    question = inputs["input"]
    history = inputs["chat_history"]

    # Identify user intent
    intent = intent_chain.invoke({
        "input": question,
        "chat_history": history
    }).strip().upper()

    print("\nDetected Intent:", intent)

    if intent == "DOCUMENT":

        return retrieve_documents(inputs)

    elif intent == "HISTORY":

        return history_answer_chain.invoke({
            "input": question,
            "chat_history": history
        })

    elif intent == "GENERAL":

        return general_chain.invoke({
            "input": question,
            "chat_history": history
        })

    else:

        return "I Don't Know based on the provided document."


rag_chain = RunnableLambda(route_request)
