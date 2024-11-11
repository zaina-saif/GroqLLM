#streamlit run /Users/zainasaif/Desktop/privacypolicies/app.py
import streamlit as st
import os
from langchain_groq import ChatGroq
# replace w ST from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load the GROQ And OpenAI API KEY 
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
#groq_api_key=os.getenv('GROQ_API_KEY')

st.title("Groq with RAG")
#openai_api_key = os.getenv("OPENAI_API_KEY")
#if openai_api_key is None:
#    raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file.")

#os.environ['OPENAI_API_KEY'] = openai_api_key

llm=ChatGroq(groq_api_key='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', model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():
    try:
        if "vectors" not in st.session_state:
            #st.session_state.embeddings=OpenAIEmbeddings()
            st.session_state.embeddings = SentenceTransformer('all-MiniLM-L6-v2')

            #load docs
            st.session_state.loader = PyPDFDirectoryLoader("/Users/zainasaif/Desktop/privacypolicies")
            st.session_state.docs = st.session_state.loader.load()
            if not st.session_state.docs:
                st.error("No documents were loaded. Please check the directory and document formats.")
                return
            
            #split
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:2])
            
            #embeddings
            document_texts = [doc.page_content for doc in st.session_state.final_documents]
            document_embeddings = st.session_state.embeddings.encode(document_texts)

           # document_embeddings = document_embeddings.cpu().numpy()
            document_embeddings = np.array(document_embeddings)

            dimension = document_embeddings.shape[1]  
            index = faiss.IndexFlatL2(dimension)  # Use L2 (Euclidean) distance
            index.add(document_embeddings)  # Add embeddings to the index

            st.session_state.vectors = {
                'index': index,
                'documents': st.session_state.final_documents
            }

    except Exception as e:
        st.error(f"An error occurred during vector embedding: {e}")
        
#fields for question 
prompt1=st.text_input("Enter Your Question")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time

if prompt1: 
    index = st.session_state.vectors['index']
    documents = st.session_state.vectors['documents']

    query_embedding = st.session_state.embeddings.encode([prompt1])
    query_embedding = np.array(query_embedding)
    
    distances, indices = index.search(query_embedding, k=5)  # Retrieve the top 5 closest documents
    for i in range(len(indices[0])):
        st.write(f"Document {i+1} (Distance: {distances[0][i]}):")
        st.write(documents[indices[0][i]].page_content)
        st.write("--------------------------------")
   # document_chain = create_stuff_documents_chain(llm, prompt)

    #retriever = st.session_state.vectors.as_retriever()
   # retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
  #  start = time.process_time()
  #  response = retrieval_chain.invoke({"input": prompt1})
  #  print("Response time:", time.process_time() - start)  # Efficiency of the RAG
    #st.write(response['answer'])

   # with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        #for i, doc in enumerate(response["context"]):
           # st.write(doc.page_content)
            #st.write("--------------------------------")