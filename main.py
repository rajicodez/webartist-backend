import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. SETUP AI MODELS
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3 # Lower temperature = more factual/professional
)

vectorstore = PineconeVectorStore(
    index_name="webartist", 
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# 2. CREATE THE PERSONA (The "Sales Engineer" Prompt)
template = """
You are the AI Growth Partner for WebArtist. You are not a robot; you are a warm, intelligent, and empathetic consultant. 
Your goal is to connect with the user's vision and explain how Webartist can help them grow.
Use the following context to answer their questions.
GUIDELINES:
1. Be warm and encouraging. Use phrases like 'We'd love to help you build that' or 'That sounds like a great vision.'
2. If asked about the team/founders/who we are, describe us EXACTLY as: 'A group of young talented individuals from various domains like Data Science, Software Engineering, Cyber Security, and Cloud Computing.' Do not mention universities or students.
3. If asked about pricing, explain that we provide customized solutions for every business, so pricing varies. Encourage them to contact us for a tailored quote.
4. Do not use complex jargon without explaining it simply.
5. Keep answers concise but friendly.

Context: {context}

Question: {question}

Helpful Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 3. CREATE CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} # <--- Inject the persona here
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    response = qa_chain.invoke(query.question)
    return {"answer": response['result']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)