
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import vectorstore


# 1. Load PDF

loader = PyPDFLoader("data/company_policy.pdf")

documents = loader.load()


# 2. Split text into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# 3. Add metadata to chunks

for chunk in chunks:

    chunk.metadata["document_type"] = "company_policy"


# 4. Store documents and embeddings in Supabase

vectorstore.add_documents(chunks)


print("Documents successfully stored in Supabase!")
