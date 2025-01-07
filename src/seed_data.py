import streamlit as st
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

def call_index(index_name: str, 
               dimention: int, 
               metric: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name,
                        dimension=dimention,
                        metric=metric,
                        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    return pc.Index(index_name)

def pinecone_hybrid_retriever(index_name: str,
                  path: str,  
                  dimentions: int = 768, 
                  metric: str = "dotproduct",
                  embeddings_name: str = "hiieu/halong_embedding") -> PineconeHybridSearchRetriever:
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    loader = DirectoryLoader(path=path, glob="**/*.docx")
    documents = loader.load()
    
    index = call_index(index_name, dimentions, metric)
    bm25_encoder = BM25Encoder().default()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    list_docs = [doc.page_content for doc in docs]
    bm25_encoder.fit(list_docs)
    bm25_encoder.dump("bm25-values.json")
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, 
                                              sparse_encoder=bm25_encoder, 
                                              index=index,
                                              alpha=0.7)
    retriever.add_text(list_docs)
    return retriever

def get_vectorstore(index_name: str, 
                    embeddings_name: str = "hiieu/halong_embedding") -> PineconeVectorStore:
    index = call_index(index_name, 768, "dotproduct")
    embeddings = HuggingFaceEmbeddings(model_name = embeddings_name)
    # bm25_encoder = BM25Encoder().load("bm25-values.json")
    # retriever = PineconeHybridSearchRetriever(index=index, embeddings=embeddings, sparse_encoder=bm25_encoder)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return vectorstore

def main():
    pinecone_hybrid_retriever('hybrid-rag', 'data', 768, 'dotproduct')
    
if __name__ == "__main__":
    main()