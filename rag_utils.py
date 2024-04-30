import glob
from dotenv import load_dotenv
load_dotenv()
from werkzeug.utils import secure_filename

# EITHER
# from langchain.document_loaders import TextLoader
# OR
# from langchain_community.document_loaders import UnstructuredPDFLoader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
# from langchain_community.document_loaders import UnstructuredPDFLoader

# LANGCHAIN
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings

# PINECONE
from pinecone import Pinecone, ServerlessSpec

# VERTEXAI
from vertexai.generative_models import GenerativeModel

# LOCAL IMPORTS
from constants import PINECONE_API_KEY

# GLOBAL VARS
pc = Pinecone(api_key=PINECONE_API_KEY)
splits_cache = {}
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
model = GenerativeModel(model_name="gemini-1.0-pro")

# CONFIGS
index_name = "langchain-demo"
dimension = 768

if index_name not in pc.list_indexes().names():
    # Pinecone does not support GCP so AWS it is on this
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(name=index_name)


# def grab_local_files(resumes: str = "", ret: bool = True):
#     """
#     grabs local resume
#     reads it for data
#     splits it into semantic chunks
#     embeds said chunks
#     uploads the vectors

#     params:
#         index -> pinecone object, the index
#         resumes -> can take a file path, if none then it grabs all resumes
#         ret -> t/f : says whether to return data or do it in place"""
#     global splits_cache
#     data = {}
#     if not resumes:
#         resumes = glob.glob("resumes/*.pdf")
#         for file_path in resumes:
#             # loader = UnstructuredPDFLoader(file_path)
#             loader = PDFMinerLoader(file_path)
#             resume_data = loader.load()
#             data[secure_filename(file_path)] = resume_data
#     else:
#         loader = PDFMinerLoader(resumes)
#         resume_data = loader.load()
#         data[secure_filename(resumes)] = resume_data


#     vectors = []
#     for file_path, contents in data.items():
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#         splits = text_splitter.split_documents(contents)

#         splits_cache[file_path] = splits

#         for i, doc in enumerate(splits):
#             doc_text = doc.page_content
#             if doc_text:
#                 try:
#                     embedding = embeddings.embed_query(doc_text)
#                     vectors.append((f"{file_path}_{i}", embedding))
#                 except Exception as e:
#                     print(f"Error embedding document from {file_path}, part {i}: {e}")

#     index.upsert(vectors=vectors)

#     if ret:
#         return data

def grab_local_files(resumes: str = "", ret: bool = True):
    global splits_cache
    data = {}
    if not resumes:
        resumes = glob.glob("resumes/*.pdf")

    for file_path in resumes:
        loader = PDFMinerLoader(file_path)
        resume_data = loader.load()
        file_key = secure_filename(file_path)
        data[file_key] = resume_data

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(resume_data)
        splits_cache[file_key] = [(file_key, split.page_content) for split in splits]

        vectors = []
        for i, (name, doc_text) in enumerate(splits_cache[file_key]):
            if doc_text:
                try:
                    embedding = embeddings.embed_query(doc_text)
                    vectors.append((f"{file_key}_{i}", embedding))
                except Exception as e:
                    print(f"Error embedding document from {file_path}, part {i}: {e}")

    index.upsert(vectors=vectors)

    if ret:
        return data


def query(question, job_title, job_desc, top_k=3):
    question_embedding = embeddings.embed_query(question)
    index_query = index.query(vector=question_embedding, top_k=top_k)
    top_docs_ids = [match["id"] for match in index_query["matches"]]

    resume_contexts = []
    for doc_id in top_docs_ids:
        print(doc_id)
        parts = doc_id.rsplit('_', 1)  # Split on the last underscore only
        if len(parts) == 2:
            file_key, split_index = parts
            split_index = split_index[0]
            print(f"Split Index: {split_index}")
            split_index = int(split_index)
            if file_key in splits_cache and split_index < len(splits_cache[file_key]):
                name, content = splits_cache[file_key][split_index]
                resume_contexts.append(f"Candidate from {name}:\n{content}")
            else:
                print(f"Missing or invalid split: {doc_id}")
        else:
            print(f"Invalid doc_id format: {doc_id}")

    resume_context = "\n\n".join(resume_contexts)
    # system_prompt = f"""
    #     You are an expert technical recruiter assistant. You specialize in vetting candidates after looking at their resume. The recruiter is looking for a candidate who can do:
    #     ```{job_desc}```
    #     You will be given a "Recruiter Question" which is the question the recruiter is asking about. You will be given "Context" which is resumes for individuals of which you will assess based on the recruiter question then you will answer the recruiter's question verbosely. Especially if you can answer with a helpful format, that would be a plus. You can assume this is just the first step in a couple sets of interviews. This step is simply finding out which candidates should move forward and which shouldn't
    # """

    system_prompt = f"""
        You are an advanced technical recruiter assistant designed to aid in the preliminary screening of candidates based on their resumes. Your role is to assess candidates' suitability for specific roles as described by the recruiter.

        For each query:
        - **Job Title**: {job_title}
        - **Job Description**: {job_desc}
        - **Recruiter Question**: This is a direct question from the recruiter regarding a candidate's fit for the role mentioned above.

        **Your task**:
        1. Review the provided "Context," which includes selected resumes.
        2. Based on the recruiter's question and the job description, evaluate which candidates should advance to the next stage of the interview process.
        3. Provide a detailed response that explains your reasoning, highlighting relevant qualifications and experiences from the resumes. Use a structured format to answer, such as listing candidates followed by bullet points of their pertinent skills or experiences.
        
        **DO NOT ASSUME OR BASE YOUR ANSWER ON GENDER, PERCEIVED RACE, SEX OR ANY POSSIBLE DEMOGRAPHIC QUALITIES A CANDIDATE MAY POSSES GIVEN WHAT YOU KNOW ABOUT THE CANDIDATES**

        **Note**: This is the first step in a multi-stage interview process. Your assessment should help narrow down the pool of candidates to those most likely to succeed in further rounds based on the job requirements.
    """

    prompt = f"{system_prompt}\nRecruiter Question: ```{question}```\n\nContext: {resume_context}\n\nAnswer:"

    answer = model.generate_content(prompt)

    answer = answer.text

    return answer


def inline_query_test(question):
    grab_local_files()
    job_title = f"""
        Intern, Machine Learning and AI Development
    """
    job_desc = f"""
        Primary Duties & Responsibilities

            Assist in the development and optimization of ML and AI models.
            Contribute to the integration and deployment of LLMs for various applications.
            Develop and maintain APIs for interacting with LLMs.
            Collaborate with the development team to design and implement new features.
            Conduct research and analysis to improve model performance and efficiency.
            Document and present findings and progress in team meetings.

        Education & Experience

            Bachelor's degree in Engineering, Computer Science, or a related field.
            Currently enrolled in advanced studies (Master's or PhD) related to Computer Science, AI, or Machine Learning.
            Proficient in Python programming.
            Strong understanding of AI, ML concepts, and algorithms.
            Experience with developing and consuming APIs.
            Excellent problem-solving and analytical skills.
            Ability to work independently and in a team environment.
            Strong communication skills, both written and verbal.

        Skills

            Prior experience with Large Language Models (e.g., OpenAI GPT) is highly desirable.
            Familiarity with software development tools and methodologies.
            Python IDE tools (Spyder, Jupyter, PyCharm etc.)

    """
    return query(question=question, job_title=job_title, job_desc=job_desc)
