import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from langchain.schema import Document


class CustomEmbeddingModel:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1"):
        self.model = SentenceTransformer(model_name,trust_remote_code=True)

    def embed_query(self, query):
        if isinstance(query, dict):  # Ensure query is a string
            query = query.get("query", "")  # Extract actual text

        if not isinstance(query, str):  # Extra safeguard
            raise ValueError("Query must be a string.")

        return self.model.encode(query).tolist()  # Convert to list if needed

# Load data and create embeddings
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    documents = []
    
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"Processing batch: {i}")

        combined_text = f"Recipe: {row['Recipe Name']}. Ingredients: {row['Ingredients']}. Instructions: {row['Instructions']}. Total Time: {row['Total Time']}. Servings: {row['Servings']}. Recipe URL: {row['Recipe URL']}"

        doc = Document(
            page_content=combined_text,  
            metadata={  
                "recipe_name": row["Recipe Name"],
                "recipe_url": row["Recipe URL"],
                "ingredients": row["Ingredients"],
                "instructions": row["Instructions"],
                "total_time": row["Total Time"],
                "servings": row["Servings"]
            }
        )
        documents.append(doc)
    
    return documents


