import langchain
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


from langchain.document_loaders.pdf import PyPDFDirectoryLoader
def load_documents():
    document_loader = PyPDFDirectoryLoader('/Users/zainasaif/Desktop/AdvancedRAG/privacypolicies2')
    return document_loader.load()

documents = load_documents() #showing whats in the doc
print(documents[0])

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len, is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)
print(chunks[0])

from langchain_community.embeddings.bedrock import BedrockEmbeddings
def get_embedding_function():
    embeddings=BedrockEmbeddings(
        credentials_profile_name="default", region_name="us-east-1"
    )
    return embeddings

#building vector db
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma
def add_to_chroma(chunks: list[Document]):
    db= Chroma(
        persist_directory=CRHOMA_PATH, embedding_function=get_embedding_function( )
    )
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()

last_page_id=None
current_chunk_index=0

for chunk in chunks:
    source = chunk.metadata.get("source")
    page=chunk.metadata.get("page")
    current_page_id=f"{source}:{page}"
    if current_page_id==last_page_id:
        current_chunk_index+=1
    else:
        current_chunk_index=0

chunk.metadata