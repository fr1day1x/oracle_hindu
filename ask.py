from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load your Gemini API key from the .env file
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# 1. Connect to the local database using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./hindu_db",
    embedding_function=embeddings
)

# 2. Define the Sage persona prompt using the new template system
system_prompt = (
    "You are a wise Hindu sage with deep knowledge of the Vedas, "
    "Upanishads, and Bhagavad Gita. When answering, draw beautifully from the "
    "sacred texts provided below. Speak with calm, profound wisdom. If the answer "
    "is present in the texts, try to cite the specific verse or passage concept.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 3. Connect to Gemini for reasoning
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GEMINI_KEY,
    temperature=0.45
)

# 4. Run the RAG process manually (Ultra-stable, zero chain dependencies!)
user_input = "What does the Gita say about the nature of the soul?"
print(f"Sending question to Oracle: {user_input}\n")

# Step A: Retrieve the top 4 text chunks matching the question from your DB
retriever = db.as_retriever(search_kwargs={"k": 4})
retrieved_docs = retriever.invoke(user_input)

# Step B: Combine those text chunks into one context block
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Step C: Format the prompt messages with our data
formatted_messages = prompt.format_messages(context=context_text, input=user_input)

# Step D: Hand it to Gemini to get the final answer
response = llm.invoke(formatted_messages)

print(f"Oracle Response:\n{response.content}")