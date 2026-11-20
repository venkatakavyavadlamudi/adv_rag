
from src.config import retriever
from src.chains import document_answer_chain

question = "What are the working hours?"

documents = retriever.invoke(question)

context = "\n\n".join(
    [doc.page_content for doc in documents]
)

answer = document_answer_chain.invoke({
    "input": question,
    "context": context
})

print("\nRetrieved Context:")
print(context)

print("\nGenerated Answer:")
print(answer)
