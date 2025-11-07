from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import the shared service instances we created earlier
from services.inference_service import inference_service
from services.data_service import data_service

# --- 1. Create the FastAPI App ---
# This is the main object that runs the whole API
app = FastAPI(
    title="Semantic NIC Code Search API",
    description="An API to find the most relevant NIC code for a business description.",
    version="1.0.0"
)

# --- 2. Configure CORS (Cross-Origin Resource Sharing) ---
# This is a critical security step that tells the browser it's okay for our
# Angular frontend (running on a different address) to communicate with this backend.
origins = [
    "http://localhost:4200",  # The default port for an Angular app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- 3. Define the Search API Endpoint ---
# This decorator tells FastAPI that the function below handles GET requests to "/api/search"
@app.get("/api/search")
def search_nic_codes(q: str = Query(..., min_length=3, description="The user's text description of the business.")):
    """
    Performs semantic search and returns the top matching NIC codes.
    """
    # Step A: Get the embedding for the user's search query.
    query_embedding = inference_service.get_embedding(q)
    
    # Step B: Get the list of all NIC records with their pre-computed embeddings.
    all_nic_records = data_service.get_all()
    
    # If data loading failed, return an error
    if not all_nic_records:
        return {"error": "Data not loaded. Please check the server logs."}

    # Step C: Prepare data for calculation.
    # We extract just the embedding vectors into a numpy array for efficient computation.
    nic_embeddings = np.array([record.embedding for record in all_nic_records])
    query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
    
    # Step D: Calculate Cosine Similarity between the query and all NIC descriptions.
    similarities = cosine_similarity(query_embedding_reshaped, nic_embeddings)[0]
    
    # Step E: Combine the records with their calculated similarity scores.
    results = []
    for i, record in enumerate(all_nic_records):
        results.append({
            "code": record.code,
            "description": record.description,
            "score": float(similarities[i]) # Convert numpy float to standard float
        })
        
    # Step F: Sort the results by score in descending order.
    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    # Step G: Return the top 10 results.
    return {"results": ranked_results[:10]}