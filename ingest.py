from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

SOURCE_DIRECTORY = 'data/'
VECTOR_STORE_PATH = 'vector_store/embedding_store'

# Function to create a vector database
def initialize_vector_database():
    document_loader = DirectoryLoader(SOURCE_DIRECTORY,
                                      glob='*.pdf',
                                      loader_cls=PyPDFLoader)

    loaded_documents = document_loader.load()
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                        chunk_overlap=50)
    split_texts = character_splitter.split_documents(loaded_documents)

    vector_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                              model_kwargs={'device': 'cpu'})

    vector_database = FAISS.from_documents(split_texts, vector_embeddings)
    vector_database.save_local(VECTOR_STORE_PATH)

if __name__ == "__main__":
    initialize_vector_database()
