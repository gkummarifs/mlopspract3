import os
import boto3
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain_aws import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# ---------- AWS Bedrock Setup ----------
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# ---------- PDF ingestion ----------
def docs_data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# ---------- Save FAISS ----------
def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")

# ---------- Load Llama2 ----------
def get_llama2_llm():
    return Bedrock(
        model_id="meta.llama2-13b-chat-v1",
        client=bedrock,
        model_kwargs={"maxTokens": 512}
    )

# ---------- QA Chain ----------
def get_response_llm(llm, vectorstore_faiss, query):
    prompt_template = """
    Human: Use the following context to answer the question at the end
    with 2-3 lines of detail. If you don't know, say you don't know.
    {context}
    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="QA with Doc")
    st.header("QA with Doc using LangChain + Bedrock")

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Processing PDFs and creating vector store..."):
            docs = docs_data_ingestion()
            get_vector_store(docs)
        st.success("Vector store created!")

    if st.button("Run with LLaMA2"):
        with st.spinner("Running LLM..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings=bedrock_embeddings)
            llm = get_llama2_llm()
            response = get_response_llm(llm, faiss_index, query)
            st.write(response["result"])
            st.success("Done")

if __name__ == "__main__":
    main()



'''import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbedding
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS

st.set_page_config(page_title="QA with Doc")
st.header("QA with Doc using lang")

query = st.text_input("Ask")

if query:
    with st.spinner("processing vectors update"):
        docs = docs_data_ingestion()
        get_vector_store(docs)
    st.success("done")
if st.button("llama model"):
    with st.spinner("processing..."):
        faiss_index = FAISS.load_local("faiss_index", embeddings=embed_embeddings)
        llm = get_llama2_llm()
        st.write(get_response_llm(llm, faiss_index, user_question))
        st.success("Done")

if __name__ == "__main__":
    main()'''
