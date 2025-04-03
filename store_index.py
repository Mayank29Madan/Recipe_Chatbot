import os
import urllib.parse
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain.vectorstores import MongoDBAtlasVectorSearch
from src.helper import CustomEmbeddingModel,load_data

# Load environment variables
load_dotenv()

MONGO_USER = urllib.parse.quote_plus(os.environ.get('MONGO_USER'))
MONGO_PASS = urllib.parse.quote_plus(os.environ.get('MONGO_PASS'))
CLUSTER_NAME = urllib.parse.quote_plus(os.environ.get('CLUSTER_NAME'))

uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@cluster0.uxade.mongodb.net/?retryWrites=true&w=majority&appName={CLUSTER_NAME}"

# Create a MongoDB client
client = MongoClient(uri, server_api=ServerApi('1'))

# Test connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. Successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Define database and collection
DB_NAME = "recipe_db"
COLLECTION_NAME = "embedded_recipes"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector-index"

db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# Load data and process embeddings
csv_path = "Data/merged-recipes.csv"  
embedding_model = CustomEmbeddingModel()
docs = load_data(csv_path)

# Insert documents into MongoDB Atlas Vector Search
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,  
    embedding=embedding_model
    collection=collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

print("Data successfully inserted into MongoDB Atlas Vector Search!")

