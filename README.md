# Semantic NIC Code Search API

This project is a high-performance web API built with Python and FastAPI that provides a semantic search engine for National Industrial Classification (NIC) codes. Instead of relying on simple keyword matching, it understands the *meaning* behind a user's text description to find the most relevant business activity code.

## Features

-   **Semantic Search:** Uses the `all-MiniLM-L6-v2` sentence-transformer model to understand the intent behind a user's query.
-   **High-Performance Backend:** Built with FastAPI and Uvicorn for fast, asynchronous request handling.
-   **Automatic API Documentation:** Provides an interactive Swagger UI at `/docs` for easy testing.
-   **Real-time Indexing:** Loads and processes the complete NIC-2008 dataset into memory on startup for incredibly fast search responses.

---

## How It Works: The Code Explained

The application's logic is designed for performance and is split into several key components.

1.  **Server Startup:**
    -   When you run `uvicorn main:app`, the application starts.
    -   The `InferenceService` (`services/inference_service.py`) is initialized **once**. It downloads the heavy `sentence-transformers` model into memory. This is our "AI Brain".
    -   The `DataService` (`services/data_service.py`) is initialized **once**. It uses the `pandas` library to read the `nic_2008_all_codes.csv` file.

2.  **The Indexing Process:**
    -   Immediately after loading the CSV, the `DataService` loops through every single NIC code description.
    -   For each description, it calls the `InferenceService` to convert the text into a numerical vector (an "embedding").
    -   It then stores the NIC code, its description, and its new embedding together in a list in memory. This pre-calculation is the key to the application's speed.

3.  **Handling a Search Request:**
    -   A user (or a frontend application) sends a GET request to `http://127.0.0.1:8000/api/search?q=a business that sells shoes`.
    -   The `search_nic_codes` function in `main.py` receives the query `"a business that sells shoes"`.
    -   It calls the `InferenceService` to generate an embedding for this specific query.
    -   It then uses `scikit-learn` to efficiently calculate the **Cosine Similarity** between the user's query embedding and *every single NIC code embedding* stored in memory.
    -   The results are ranked by their similarity score (from highest to lowest).
    -   The top 10 best matches are returned as a JSON response.

---

## Project Structure

```
SemanticSearchApiPython/
├── services/
│   ├── data_service.py       # Loads, indexes, and holds all NIC data in memory.
│   └── inference_service.py    # The "AI Core": loads the model and generates embeddings.
├── main.py                     # The main FastAPI file: defines the API endpoint and server logic.
├── nic_2008_all_codes.csv      # The raw NIC-2008 dataset.
├── requirements.txt            # A list of all required Python libraries.
└── README.md                   # This file.
```

---

## Setup and Running Instructions (for Windows)

Follow these steps to get the API running on your local machine.

### Prerequisites

-   You must have Python 3.8+ installed on your system.

### Step 1: Create and Activate the Environment

Open your terminal (PowerShell or Command Prompt), navigate to this project folder, and run the following commands.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
.\venv\Scripts\activate
```
*(Note: If you get a security error in PowerShell, you may need to run `Set-ExecutionPolicy RemoteSigned` in an Administrator PowerShell first.)*

### Step 2: Install Dependencies

Install all the required libraries using the `requirements.txt` file. This command will read the file and install the exact versions needed.

```bash
pip install -r requirements.txt
```

### Step 3: Run the API Server

Start the application using the Uvicorn server.

```bash
uvicorn main:app --reload
```

The first time you run this, `sentence-transformers` will download the AI model (approx. 90MB), which may take a minute. After that, the server will start, load the CSV data, and index it.

You will see a line in the terminal that says: `Uvicorn running on http://127.0.0.1:8000`.

### Step 4: Test the API

1.  Open your web browser and navigate to: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2.  Click the `GET /api/search` endpoint to expand it.
3.  Click the **"Try it out"** button.
4.  Enter a business description in the `q` value box (e.g., `a company that sells clothes and shoes` or `making custom software for computers`).
5.  Click **"Execute"**.

You will see the top 10 JSON results ranked by their relevance score.