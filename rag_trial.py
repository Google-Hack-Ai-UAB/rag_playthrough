# Adapt the following tutorials to use Vertex AI embeddings
# from https://python.langchain.com/docs/integrations/vectorstores/pinecone
# and https://python.langchain.com/docs/use_cases/question_answering/


import os
import getpass
from dotenv import load_dotenv
import vertexai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone

# EITHER
# from langchain.document_loaders import TextLoader
# OR
from langchain_community.document_loaders import UnstructuredPDFLoader
# OR
# from langchain_community.document_loaders import PDFMinerLoader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

from pinecone import Pinecone, ServerlessSpec
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import VertexAI
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

project_id = "vertexaiconversations-418821"
location = "us-central1"
project_num = 316665376175

vertexai.init(project=project_id, location=location)


loader = UnstructuredPDFLoader("resumes/resume.pdf")
#  UnstructuredPDFLoader("resumes/resume.pdf", mode="elements")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(loader.load())
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "langchain-demo"
dimension = 768

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region="us-east-1")
    )


index = pc.Index(name=index_name)

# vectors = [(str(i), embeddings.embed_query(doc)) for i, doc in enumerate(splits)]
vectors = [(str(i), embeddings.embed_query(str(doc))) for i, doc in enumerate(splits)]
vectors = []
for i, doc in enumerate(splits):
    doc_text = doc.page_content  
    if doc_text:
        try:
            embedding = embeddings.embed_query(doc_text)
            vectors.append((str(i), embedding))
        except Exception as e:
            print(f"Error generating embedding for document {i}: {e}")

index.upsert(vectors=vectors)

# Prompt 
# https://smith.langchain.com/hub/rlm/rag-prompt


rag_prompt = hub.pull("rlm/rag-prompt")

print("past index rag_prompt creation ")


def retrieve_top_documents(question, index, embeddings, top_k=5):
    question_embedding = embeddings.embed_query(question)
    
    query_results = index.query(vector=question_embedding, top_k=top_k)
    
    top_documents = [(match["id"], match["score"]) for match in query_results["matches"]]
    
    return top_documents

question = "Do you think Michael Gathara will be a good fit for a position that requires knowing Javascript?"
top_documents = retrieve_top_documents(question, index, embeddings, top_k=5)
print("Top documents:", top_documents)


from vertexai import generative_models
from vertexai.generative_models import GenerativeModel
model = GenerativeModel(model_name="gemini-1.0-pro")

def build_context_from_ids(document_ids, splits):
    context_texts = [splits[int(doc_id)].page_content for doc_id in document_ids]
    return " ".join(context_texts)


def rag_query(question, index, embeddings, llm_model, top_k=5):
    # Retrieve top document IDs from Pinecone
    question_embedding = embeddings.embed_query(question)
    query_results = index.query(vector=question_embedding, top_k=top_k)
    top_documents_ids = [match["id"] for match in query_results["matches"]]

    context = build_context_from_ids(top_documents_ids, splits)

    prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"

    answer = llm_model.generate_content(prompt)
    
    answer = answer.text
    
    return answer


answer = rag_query(question, index, embeddings, model, top_k=5)
print("Generated answer:", answer)


# RAG chain 

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()} 
#     | rag_prompt 
#     | llm 
# )

# rci_output = rag_chain.invoke("Who is Mitchell Kimbell")
# print("rci_output is: ",rci_output)


