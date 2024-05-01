# rag_playthrough

## How to Run

make env
`python3 -m venv env/`
`source env/bin/activate`

install requirements
`pip install -r requirements.txt`

## Tools Used
* **Pinecone**: Serverless vector database used for storage of candidate information
* **Gemini**: Google's AI model used for human readable suggestions and conversation based on PDF submissions
* **VertexAI**: Google's machine learning platform used for sending requests to Gemini with embedded PDF data
* **MongoDB**: NoSQL database used for raw candidate resume PDF storage

