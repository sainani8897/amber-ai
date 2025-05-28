# prepare_data.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set your OpenAI API key (replace with your actual key or use environment variable)
# It's highly recommended to use environment variables for API keys in production
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define the directory where your Chroma vector store will be persisted
PERSIST_DIRECTORY = 'chroma_db'
DATA_FILE = 'business_data.txt'

def prepare_business_data():
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Vector store already exists. Skipping data preparation.")
        return

    print(f"Loading data from {DATA_FILE}...")
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings and building vector store...")
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Create a Chroma vector store from the chunks and persist it
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    print(f"Vector store created and persisted to {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    # Create a dummy data file for demonstration if it doesn't exist
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w") as f:
            f.write("""Our company, "Tech Innovations Inc.", was founded in 2005. We specialize in developing cutting-edge AI solutions for enterprises. Our main products include the "AI-Assistant Pro" for customer service automation and "Data Insight Engine" for business intelligence. We offer 24/7 customer support via email at support@techinnovations.com or phone at +1-800-TECH-AI. Our headquarters are located in Silicon Valley, California.
            We have a strict return policy of 30 days for all physical products, provided they are in their original packaging and condition. Digital products are non-refundable. For service cancellations, please contact customer support at least 48 hours in advance.
            Our pricing structure for "AI-Assistant Pro" starts at $199/month for the basic plan, and $499/month for the enterprise plan which includes dedicated support and custom integrations. Discounts are available for annual subscriptions.
            """)
        print(f"Created dummy {DATA_FILE} for demonstration.")

    prepare_business_data()