
# ingestion.py
import os
import time
import random
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

# ----------------------------
# Bedrock Client Setup
# ----------------------------
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", 
    client=bedrock
)

# ----------------------------
# Helper: Retry with Backoff
# ----------------------------
def safe_embed(texts, retries=5):
    for attempt in range(retries):
        try:
            return bedrock_embeddings.embed_documents(texts)
        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"[Retry {attempt+1}/{retries}] Error: {e}. Waiting {wait:.2f}s...")
            time.sleep(wait)
    raise RuntimeError("Failed after max retries")


# ----------------------------
# Step 1: Data Ingestion
# ----------------------------
def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")   # Make sure ./data has your PDFs
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    print(f"✅ Loaded {len(docs)} chunks from PDFs")
    return docs


# ----------------------------
# Step 2: Create FAISS Vector Store with Batching
# ----------------------------
def get_vector_store(docs, batch_size=10):
    all_stores = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        print(f"🔹 Embedding batch {i//batch_size + 1} with {len(batch)} docs...")

        # Use safe embed for retries
        texts = [doc.page_content for doc in batch]
        embeddings = safe_embed(texts)

        # Build FAISS from this batch
        batch_store = FAISS.from_texts(texts, bedrock_embeddings, metadatas=[doc.metadata for doc in batch])
        all_stores.append(batch_store)

    # Merge all batches into one FAISS index
    final_store = all_stores[0]
    for store in all_stores[1:]:
        final_store.merge_from(store)

    final_store.save_local("faiss_index")
    print("✅ FAISS index saved locally as 'faiss_index'")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs, batch_size=10)


'''from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain_aws import BedrockEmbeddings

import json
import os
import sys
import boto3

##Bedrock client

bedrock=boto3.client(service_name= "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=1000)
    text_splitter.split_documents(documents)
    docs = text_splitter.split_documents(documents)
    
    return docs
    # print(f"Length of documents {len(docs)}")
    
def get_vector_store(docs):
    vector_store_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss


if __name__ == "__main__":
    docs=data_ingestion() 
    get_vector_store(docs)'''
    
    
