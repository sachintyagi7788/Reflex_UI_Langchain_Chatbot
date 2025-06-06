
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings



def create_qa_chain(pdf_path: str):
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Increased overlap
    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError("No text could be extracted from the PDF or the PDF is empty.")

    print("Creating embeddings and vector store...")
   
    vectorstore = FAISS.from_documents(texts, embeddings)

    print("Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm= llm,# llm_config.llm,  #ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo"), # Specify model
        chain_type="stuff", # "stuff" is good for smaller contexts
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
        return_source_documents=True # Optionally return source documents
    )
    print("QA chain created successfully.")
    return qa_chain

async def get_answer_from_chain(qa_chain, question: str):
    
    if not qa_chain:
        return "QA chain is not initialized."
    print(f"Querying chain with: {question}")
  
    response = await qa_chain.ainvoke({"query": question}) # Pass input as a dictionary
    print(f"Chain response: {response}")
    return response.get("result", "No answer found.")
    