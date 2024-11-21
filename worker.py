from typing import List, Tuple
import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

# Check for GPU availability and set the appropriate device for computation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history: List[Tuple[str, str]] = []
llm = None
embeddings = None


def init_llm(model_name: str = "mistral"):
    """
    Initialize the local LLM using Ollama

    Args:
        model_name (str): Name of the Ollama model to use
    """
    global llm, embeddings

    # Initialize Ollama LLM
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,  # Lower temperature for more focused responses
        num_predict=600,  # Maximum number of tokens to generate
    )

    # Initialize embeddings (switched to standard HuggingFaceEmbeddings)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )


def process_document(document_path: str):
    """
    Process a PDF document and create a retrieval chain

    Args:
        document_path (str): Path to the PDF file
    """
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # Create an embeddings database using Chroma
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Build the QA chain
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}),
        return_source_documents=False,
        input_key="question",
    )


def process_prompt(prompt: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process a user prompt and return a response with chat history.

    Args:
        prompt (str): User's input message

    Returns:
        Tuple[str, List[Tuple[str, str]]]: Model's response and updated chat history
    """
    global conversation_retrieval_chain, chat_history

    if conversation_retrieval_chain is None:
        return "Please upload and process a PDF document first.", chat_history

    try:
        output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
        answer = output["result"]
        chat_history.append((prompt, answer))
        return answer, chat_history
    except Exception as e:
        return f"An error occurred: {str(e)}", chat_history


# Initialize the language model when the module is imported
if __name__ == "__main__":
    init_llm()
