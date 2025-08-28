# query.py
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock
from QASystem.ingestion import get_vector_store, data_ingestion

import boto3

# ----------------------------
# Bedrock client
# ----------------------------
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# ----------------------------
# Prompt Template
# ----------------------------
prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end. Provide at least
2–3 words with detailed explanations. If you don't know the answer,
just say that you don't know — do not make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ----------------------------
# LLM Loader
# ----------------------------
def get_llama2_llm():
    llm = Bedrock(
        model_id="meta.llama2-13b-chat-v1",
        client=bedrock,
        model_kwargs={"maxTokens": 512}
    )
    return llm

# ----------------------------
# QA Function
# ----------------------------
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": query})
    return answer

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    docs = data_ingestion()
    vectorstore_faiss = get_vector_store(docs)   # Make sure ingestion returns FAISS index
    query = "What is RAG token?"
    llm = get_llama2_llm()

    response = get_response_llm(llm, vectorstore_faiss, query)
    print("Answer:", response["result"])
    print("Sources:", [doc.metadata for doc in response["source_documents"]])



'''from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import boto3
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from QASystem.ingestion import get_vector_store
from QASystem.ingestion import data_ingestion

bedrock=boto3.client(service_name="bedrock-runtime")

prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end but use at least
2-3 words with detailed explanations. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
{context}
</context>

Question: {question}

Assistant:""" #Rsponse/Answer

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]   
    
)

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client=bedrock, model_kwargs={"maxTokens":512})
    return llm




def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": query})
    return answer

if __name__ == '__main__':
    docs=data_ingestion()
    vectorstore_faiss = get_vector_store(docs)
    query = "What is RAG token?"
    llm = get_llama2_llm()
    get_response_llm(llm,query)
    
'''
