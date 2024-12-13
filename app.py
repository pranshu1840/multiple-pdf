import os
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chat_models import ChatOpenAI

load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME")
index = pc.Index(index_name)

# Define HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Load documents
loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Generate the embeddings for the documents
document_embeddings = embeddings.embed_documents([chunk.page_content for chunk in text_chunks])

# Upsert the vectors to Pinecone (index the embeddings)
def upload_data(index, embeddings, text_chunks):
    vectors = []
    for i, embedding in enumerate(embeddings):
        vectors.append((str(i), embedding, {"page_content": text_chunks[i].page_content}))
    try:
        index.upsert(vectors=vectors)
        print("Data uploaded successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Upload the data (vectors)
upload_data(index, document_embeddings, text_chunks)

# Now, create the Pinecone vector store for LangChain (correct usage)
# 'text_key' specifies which field contains the text data
vector_store = LangChainPinecone(index, embeddings.embed_query, text_key="page_content")

# Load the model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Query the Supabase database for the username
        user = supabase.table("users").select("*").eq("username", username).execute()

        if user.data and user.data[0]['password'] == password:  # Simple password check
            session["user"] = user.data[0]
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        # Check if username already exists
        existing_user = supabase.table("users").select("*").eq("username", username).execute()
        
        if existing_user.data:
            return render_template("register.html", error="Username already exists")
        
        # Register new user
        new_user = {
            "username": username,
            "password": password  # Store password securely, e.g., hashing is recommended
        }
        
        # Insert into Supabase
        supabase.table("users").insert(new_user).execute()
        
        # Automatically log the user in after registration
        session["user"] = new_user
        return redirect(url_for("index"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    result = chain({"question": user_input, "chat_history": []})
    return result["answer"]

@app.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if "pdf-files" not in request.files:
        return "No file part"

    files = request.files.getlist("pdf-files")
    for file in files:
        file.save(os.path.join("data", file.filename))

    # Re-load documents and re-index after new uploads
    loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    document_embeddings = embeddings.embed_documents([chunk.page_content for chunk in text_chunks])

    # Upsert the new vectors to Pinecone
    upload_data(index, document_embeddings, text_chunks)

    return "PDFs uploaded and vectors indexed successfully!"

if __name__ == "__main__":
    app.run(debug=True)
