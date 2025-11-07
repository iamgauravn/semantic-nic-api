from sentence_transformers import SentenceTransformer

class InferenceService:
    """
    A service to handle the AI model loading and embedding generation.
    We create it as a class to ensure the heavy model is loaded only ONCE.
    """
    def __init__(self):
        # The model is downloaded from the internet and cached automatically 
        # the first time you run this. This is a heavy object, so we only
        # want to create it one time when the application starts.
        print("Loading AI model... This may take a moment on first run.")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("AI model loaded successfully.")

    def get_embedding(self, text: str) -> list[float]:
        # The .encode() method handles the entire complex pipeline for us:
        # 1. Tokenization (splitting text into pieces the model understands)
        # 2. Inference (running the text through the AI model)
        # 3. Pooling (creating a single vector for the whole sentence)
        # 4. Normalization (making the vector ready for comparison)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

# --- SINGLETON INSTANCE ---
# We create a single, shared instance of the service right here.
# Other parts of our app will import this one instance instead of creating new ones.
# This is the key to good performance.
inference_service = InferenceService()