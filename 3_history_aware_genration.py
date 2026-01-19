from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

load_dotenv()
 
persistant_directory= 'db/chroma_db'

#load embedding model and vector store
embedding_model = OpenAIEmbeddings(model = 'text-embedding-3-small')

db= Chroma(
    persist_directory=persistant_directory,
    embedding_function=embedding_model,
    collection_configuration={"hnsw:space":"cosine"}
)

model = ChatOpenAI(model = 'gpt-4o')

chat_history = []

def ask_question(question):
    print(f"\n you asked : {question}")

    if chat_history:
        messages = [
            SystemMessage(content="given the chat history, rewrite the new question to be standalone and searchable. just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"new question : {question}")
        ]

        results = model.invoke(messages)
        search_question = results.content.strip()
        print(f"searching for:{search_question}")

    else:
        search_question=question

    # find relevant documents
    retrieval = db.as_retriever(search_kwargs={"k":5})
    docs = retrieval.invoke(search_question)            

    print(f"found {len(docs)} relevent documents")
    for i,doc in enumerate(docs,1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"doc {i}:{preview}...")


    #create final prompt
    combined_input = f""" Based on the following documents, please answer this uestion{question},
    Documents:
    {'\n'.join([f"-{doc.page_content}" for doc in docs])}

    please provide clear,helpful asnwers only the information from these documents. if you cant find information then return "sorry, i cant help you with this question as it is out of my knowledge base"
    """
    messages = [
        SystemMessage(content=" you are a helpfull assistant that answers the question based on the provided documents and conversations")    
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    result = model.invoke(messages)
    answer = result.content
    chat_history.append(HumanMessage(content=combined_input))
    chat_history.append(AIMessage(content=answer))

    print(f"answer:{answer}")
    return answer


def start_chat():
    print("ask me questions! type 'exit' to quit")

    while True:
        question = input('\n your question: ')

        if question.lower() == 'exit':
            print("exiting chat...")
            break
        ask_question(question)

if __name__ == "__main__":
    start_chat()