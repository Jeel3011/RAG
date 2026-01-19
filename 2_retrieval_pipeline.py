from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()
 
persistant_directory= 'db/chroma_db'

#load embedding model and vector store
embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')

db= Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_configuration={"hnsw:space":"cosine"}
)

#search for relevant documents

query = "what was NVIDIA's first graphics accelaerator called?" 

retrieval = db.as_retriever(search_kwargs={"k":5})


relevant_docs = retrieval.invoke(query)

print(f"user query: {query}\n")

print("context retrieved from vector store: \n")
for i,doc in enumerate(relevant_docs):
    print(f" Document {i}:\n {doc.page_content}\n")

#generate answer using retrieved documents as context

combined_input = f"""based on the following documents , please answer this question : {query}\n\n

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

please provide clear, helpful answer using information from the documents. if you cant find the answer then reply with "no relevant information found in the documents."

"""

model=ChatOpenAI(model = 'gpt-4o')

message=[
    HumanMessage(content=combined_input),
    SystemMessage(content="you are a helpful assistant.")
]

result = model.invoke(message)

print("\n generated response:\n")

print(result.content)