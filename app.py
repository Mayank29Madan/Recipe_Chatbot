from flask import Flask, render_template, jsonify, request
from src.helper import CustomEmbeddingModel
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from huggingface_hub import login
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from src.prompt import *
from pyngrok import ngrok
import threading
import time
import os
import urllib
import sys
import atexit

def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()


app = Flask(__name__)
load_dotenv()

HF_TOKEN = urllib.parse.quote_plus(os.environ.get('HF_TOKEN'))
MONGO_USER = urllib.parse.quote_plus(os.environ.get('MONGO_USER'))
MONGO_PASS = urllib.parse.quote_plus(os.environ.get('MONGO_PASS'))
CLUSTER_NAME = urllib.parse.quote_plus(os.environ.get('CLUSTER_NAME'))

# Login to Hugging Face
login(token=HF_TOKEN)  # Replace with your actual token

# Create the directory if it doesn't exist
os.makedirs("IP/Model", exist_ok=True)

# Download the model to the specified directory
model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    local_dir="Model"
)

print(f"Model downloaded to: {model_path}")

MONGO_URI = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@cluster0.uxade.mongodb.net/?retryWrites=true&w=majority&appName={CLUSTER_NAME}"

DB_NAME = "recipe_db"
COLLECTION_NAME = "embedded_recipes"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

model = CustomEmbeddingModel()

#def format_document(doc):
#   return f"Recipe: {doc['recipe_name']}\nIngredients: {doc['ingredients']}\nInstructions: {doc['instructions']}"

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
   MONGO_URI,
   f"{DB_NAME}.{COLLECTION_NAME}",
   model,
   index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
   text_key="recipe_name"
)

qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 200,
        "post_filter_pipeline": [{"$limit": 25}]
    }
)

# Test the retriever manually
#test_results = qa_retriever.invoke({"query": "Biryani Recipe"})
#print("Retrieved Documents:", test_results)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=['context', 'input']
)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="Model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=100,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    # Adding these parameters for better performance
    n_ctx=2048,  # Context window
    n_batch=512,  # Batch size for prompt processing
    f16_kv=True,  # Use half-precision for key/value cache
    streaming=True,  # Enable streaming for faster response time
)

# Use RefineDocumentsChain explicitly instead of load_qa_chain
combine_docs_chain = create_stuff_documents_chain(
    llm, PROMPT
)

# Define the final QA chain
qa = create_retrieval_chain(qa_retriever, combine_docs_chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip() or request.json.get("msg", "").strip()

    #print("Received input:", repr(msg))
    if not msg:
        return "Please send a message!", 400

    query_input = {"input": str(msg)}
    #print("Final Input to RetrievalQA:", query_input)

    try:
        # Invoke the chain and get result
        result = qa.invoke(query_input)
        
        # Extract the response text
        if isinstance(result, dict):
            response = result.get("result", result.get("answer", ""))
        else:
            response = str(result)

        # Ensure we have a string response
        response = str(response) if response else "I apologize, but I couldn't generate a proper response."
        
        # Clean the response to handle any special characters
        response = response.replace('\n', '<br>')
        
        print("Response:", response)
        return response

    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}", 500

def run_flask():
    app.run(host="0.0.0.0", port=5000)

def close_model():
    global llm
    if llm is not None:
        print("Closing Llama model...")
        del llm

atexit.register(close_model)

if __name__ == '__main__':
    try:
        public_url = ngrok.connect(5000).public_url
        print(f"Public URL: {public_url}")
        
        threading.Thread(target=run_flask, daemon=True).start()
        
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        close_model()
    except Exception as e:
        print(f"Error: {str(e)}")
        close_model()