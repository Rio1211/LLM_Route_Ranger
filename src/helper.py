from typing import List

try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, DirectoryLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

#Extract Data from the PDF file
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*pdf",
                             loader_cls=PyPDFLoader)
    documents=loader.load()

    return documents

def filter_text(docs: List[Document]) -> List[Document]:
    """
    Given a list of documents, 
    return a list of documents where the text has been filtered out
    with only 'source' in metadata and original page_content.
    """

    filter_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        filter_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return filter_docs

def text_split(filter_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,   #tokens
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(filter_docs)
    return texts_chunk

def download_embeddings():
    """
    Download and return the HuggingFace Embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name)
    return embeddings
