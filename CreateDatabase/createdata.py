from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import dotenv_values

config = dotenv_values(".env")

os.environ['OPENAI_API_KEY'] = config['openai_api']

tarPath = "./Doc"

def load_document():
    documents = []
    for root, dirs, files in os.walk(tarPath):
        for file in files:
            full_path = os.path.join(root, file)
            print (full_path)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(full_path)
                documents.extend(loader.load())
            elif (file.endswith('.txt')):
                # Prepend the filename to the document content
                file_header = f"--- BEGIN FILE: {file} in {full_path} "
                if(file.endswith(".txt")):
                    file_header +="which is text file.---\n"
                else :
                    file_header +="\n"
                #donot forget to append to chromadb
                file_footer = f"\n--- END FILE: {file} ---"
                loader = TextLoader(full_path)
                documents.extend(loader.load())
    return documents

def split_text(documents):
    text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=950, chunk_overlap=450, length_function=len, add_start_index=True,
    )
    chunks = text_spliter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chuncks.") 
    # print(chunks)
    documents = chunks[0]
    print(documents)
    print(documents.page_content)
    print(documents.metadata)
    return chunks

def save_to_vector(chuncks):
    data = "./Data/"
    vectordb = Chroma.from_documents(chuncks, embedding=OpenAIEmbeddings(), persist_directory=data)
    vectordb.persist()
    print(f"Saved {len(chuncks)} chunks to {data}.")


if __name__ == "__main__":
    docs = load_document()
    chunks = split_text(docs)
    vectorDB = save_to_vector(chunks)