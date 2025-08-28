from setuptools import find_packages, setup

setup(
    name="qasystem",
    version="0.0.1",
    author="Govind Kummari",
    author_email="gkummari@gmail.com",
    packages=find_packages(),
    install_requires=["langchainhub","bs4","tiktoken","boto3==1.34.37","langchain_community","chromadb","awscli","streamlit","pypdf","faiss-cpu"]
       
)
