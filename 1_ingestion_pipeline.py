import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#loading documents from the specified directory
def load_docs(docs_path='docs'):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'The specified documents path {docs_path} does not exist.')

    loader = DirectoryLoader(
        path = docs_path,
        glob='*.txt',       ### only load .txt files
        loader_cls=TextLoader
    )
    documents = loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f'No documents found in {docs_path}.')
    
    for i,doc in enumerate(documents[:2]):
        print(f"\n Document {i+1}:")
        print(f" Source : {doc.metadata['source']}")
        print(f" Content length : {len(doc.page_content)} characters")
        print(f" Content preview : {doc.page_content[:200]}...")
        print(f" metadata : {doc.metadata}")

    return documents    


# splitting documents into smaller chunks
def split_docs(documents,chunk_size=1000,chunk_overlap=200):
    print("\n\nsplitting documents into chunks...\n")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"\nTotal chunks created: {len(chunks)}")

    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1}---")
            print(f" Content length : {len(chunk.page_content)} characters")
            print(f" Content preview : {chunk.page_content[:200]}...")
            print(f" metadata : {chunk.metadata}")
            print(chunk.page_content[:200])
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... {len(chunks)-5} more chunks not displayed ...")
    return chunks



# creating vector store from document chunks
def create_vector_store(chunks,persist_directory='db/chroma_db'):
    print("\n\n creating vectoe store,,,\n")

    emmbedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')
    
    print(("creating vector store with chroma..."))

    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=emmbedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}

    )

    print("finalyzing vector store...")
    print(f"vector store created at {persist_directory}")
    return vectorstore



def main():
    
    documents = load_docs(docs_path='docs')

    chunks = split_docs(documents=documents)

    vecorestore= create_vector_store(chunks=chunks)
if __name__ == "__main__":
    main()