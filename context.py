import gdown
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Iterable, Optional

def download_context_data(pdfs: Iterable[dict[str, str]], path: str = "./context_data"):
    """
    Downloads PDFs and stores them in local storage.
    
    :param pdfs: an iterable of dictionaries. each dictionary must have
                 keys of "url" with the URL of the PDF and "filename"
                 with the name to store the file as.
    :param path: location to store the files (default is "./context_data")
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        gdown.download(pdf["url"], f"{path}/{pdf['filename']}", quiet=True)

def load_context_data(path: str = "./context_data"):
    """
    Loads multiple PDFs into LangChain Document objects.
    
    :param path: location of the files (default: "./context_data")
    :returns: list of Document objects
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def chunk_context_data(context_data: list[Document]):
    """
    Split the context data into overlapping chunks
    
    :param context_data: An iterable of Documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents(context_data)

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Gets an embedding model for vectorizing the context data.

    :param model_name: The name of the embedding model to get (default is sentence-transformers/all-MiniLM-L6-v2)
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def get_vector_store(chunks: Optional[list[Document]] = None, embedding_model: Optional[Embeddings] = None, path: str = "./chromadb"):
    """
    Get a persistent vector store. If it doesn't exist, create it from the provided context data.
    
    :param chunks: The context data
    :param embedding_model: The embedding model to use (default is the default for get_embedding_model)
    :param path: Path to create the vector store

    :returns: The vector store
    """
    if not embedding_model:
        embedding_model=get_embedding_model()

    if Path(path).exists():
        return Chroma(
            persist_directory=path,
            embedding_function=embedding_model
        )
    else:
        return Chroma.from_documents(
            documents=chunks or [],
            embedding=embedding_model,
            persist_directory=path
        )

if __name__ == "__main__":
    # when run as a script, run some tests to demonstrate capabilities
#    pdfs = (
#        { "url": "https://quanticedu.github.io/praxa/Longest Running Shows on Broadway 2025.pdf",
#          "filename": "Longest Running Shows on Broadway.pdf" },
#        { "url": "https://quanticedu.github.io/praxa/Every play and musical coming to the West End in 2025.pdf",
#          "filename": "Every play and musical coming to the West End in 2025.pdf" }
#    )
#    download_context_data(pdfs)
#    context_data = load_context_data()
#    chunks = chunk_context_data(context_data)
#    embedding_model = get_embedding_model()
#    vector_store = get_vector_store(chunks, embedding_model)

#    for page in context_data:
#        print(page)

#    for num, chunk in enumerate(chunks):
#        print("-----")
#        print(f"Chunk {num}:")
#        print(f"Length: {len(chunk.page_content)}")
#        print(f"Metadata: {chunk.metadata}")
#        print(f"Content: {chunk.page_content}")

#    embedding = embedding_model.embed_query("This is a test sentence.")
#    print(f"Embedding length: {len(embedding)}")
#    embedding = embedding_model.embed_query("This is a longer test sentence.")
#    print(f"Embedding length: {len(embedding)}")
    
#    retrieved_chunks = vector_store.similarity_search("A play written by Ryan Calais Cameron.")
#    print(f"Query retrieved {len(retrieved_chunks)} chunks.")

#    for chunk in retrieved_chunks:
#        print(f"Chunk content: {chunk.page_content}")
#        print(f"Chunk metadata: {chunk.metadata}")
#        print("-----")

    pass