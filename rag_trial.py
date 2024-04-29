"""
THIS FILE IS EXTREMELY BROKEN, WILL NOT RUN, THIS IS JUST STORAGE, WILL BE DELETED LATER
"""



# Adapt the following tutorials to use Vertex AI embeddings
# from https://python.langchain.com/docs/integrations/vectorstores/pinecone
# and https://python.langchain.com/docs/use_cases/question_answering/


import os, timeit
from dotenv import load_dotenv
import vertexai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

# from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone

# EITHER
# from langchain.document_loaders import TextLoader
# OR
# from langchain_community.document_loaders import UnstructuredPDFLoader

# OR
# from langchain_community.document_loaders import PDFMinerLoader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf

from pinecone import Pinecone, ServerlessSpec
# from langchain import hub
# from langchain.prompts import ChatPromptTemplate
# from langchain_google_vertexai import VertexAI
# from langchain.schema.runnable import RunnablePassthrough

# from vertexai import generative_models
from vertexai.generative_models import GenerativeModel

# LOCAL IMPORTS
from constants import GCP_LOCATION, GCP_PROJECT_ID, GCP_PROJECT_NUM
from rag_utils import *

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vertex_init_start = timeit.default_timer()
vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
vertex_init_stop = timeit.default_timer()
print(f"Vertex init: {vertex_init_stop - vertex_init_start}")

model_gen_start = timeit.default_timer()
model = GenerativeModel(model_name = "gemini-1.0-pro")
model_gen_end = timeit.default_timer()
print(f"Model generation time: {model_gen_end - model_gen_start}")

# loader = UnstructuredPDFLoader("resumes/resume.pdf")
#  UnstructuredPDFLoader("resumes/resume.pdf", mode="elements")
# data = loader.load()
data = grab_local_files()

# text_splitting_start = timeit.default_timer()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# # splits = text_splitter.split_documents(loader.load())
# splits = text_splitter.split_documents(data)
# embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
# text_splitting_end = timeit.default_timer()
# print(f"Text splitting: {text_splitting_end - text_splitting_start}")


# index_name = "langchain-demo"
# dimension = 768

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=dimension,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# index = pc.Index(name=index_name)

# vectors = [(str(i), embeddings.embed_query(doc)) for i, doc in enumerate(splits)]
# vectoring_start = timeit.default_timer()
# vectors = [(str(i), embeddings.embed_query(str(doc))) for i, doc in enumerate(splits)]
# vectors = []
# for i, doc in enumerate(splits):
#     doc_text = doc.page_content
#     if doc_text:
#         try:
#             embedding = embeddings.embed_query(doc_text)
#             vectors.append((str(i), embedding))
#         except Exception as e:
#             print(f"Error generating embedding for document {i}: {e}")

# index.upsert(vectors=vectors)

# vectors = []
# for i, doc in enumerate(splits):
#     doc_text = getattr(doc, 'page_content', "")
#     if doc_text:
#         try:
#             embedding = embeddings.embed_query(doc_text)
#             vectors.append((str(i), embedding))
#         except Exception as e:
#             print("ERROR")
# index.upsert(vectors=vectors)
# vectoring_end = timeit.default_timer()
# print(f"Vectoring time: {vectoring_end - vectoring_start}")
# Prompt
# https://smith.langchain.com/hub/rlm/rag-prompt


# rag_prompt = hub.pull("rlm/rag-prompt")

print("past index rag_prompt creation ")


def retrieve_top_documents(question, index, embeddings, top_k=5):
    retrieval_start = timeit.default_timer()

    question_embedding = embeddings.embed_query(question)

    query_results = index.query(vector=question_embedding, top_k=top_k)

    top_documents = [
        (match["id"], match["score"]) for match in query_results["matches"]
    ]

    retrieval_end = timeit.default_timer()
    print(f"Retrieval Time: {retrieval_end - retrieval_start}")

    return top_documents


def build_context_from_ids(document_ids, splits):
    context_texts = [splits[int(doc_id)].page_content for doc_id in document_ids]
    return " ".join(context_texts)


def rag_query(question, index, embeddings, llm_model, top_k=5):
    # Retrieve top document IDs from Pinecone
    question_embedding = embeddings.embed_query(question)
    query_results = index.query(vector=question_embedding, top_k=top_k)
    top_documents_ids = [match["id"] for match in query_results["matches"]]

    context = build_context_from_ids(top_documents_ids, splits)

    # system_prompt = "You are an incredible recruiter assister. You to help a recruiter in finding the best candidate for a position. You are given a question, posed by a recruiter, then given context which may be information about candidates who applied to the position and you are to be somewhat verbose in your answer to be helpful. You can assume that the recruiter will conduct a technical interview as the next step. You are simply here to help assess if the applicants will be good for the position\n"

    job_desc = ""

    system_prompt = f"""
        You are an expert technical recruiter assistant. You specialize in vetting candidates after looking at their resume. The recruiter is looking for a candidate who can do: 
        ```{job_desc}```
        You will be given a "Recruiter Question" which is the question the recruiter is asking about. You will be given "Context" which is resumes for individuals of which you will assess based on the recruiter question then you will answer the recruiter's question verbosely. Especially if you can answer with a helpful format, that would be a plus. You can assume this is just the first step in a couple sets of interviews. This step is simply finding out which candidates should move forward and which shouldn't
    """

    prompt = f"{system_prompt}\nRecruiter Question: ```{question}```\n\nContext: {context}\n\nAnswer:"

    answer = llm_model.generate_content(prompt)

    answer = answer.text

    return answer


question_start = timeit.default_timer()
question = "Do you think Michael Gathara will be a good fit for a position that requires knowing Javascript?"
top_documents = retrieve_top_documents(question, index, embeddings, top_k=5)

answer = rag_query(question, index, embeddings, model, top_k=5)
question_end = timeit.default_timer()

print(f"Time taken: {question_end - question_start} seconds")
print("Generated answer:", answer)
