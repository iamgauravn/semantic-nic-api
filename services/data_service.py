# We import the single instance of our inference service from the other file
from .inference_service import inference_service
import pandas as pd   
import os            

class NicCodeRecord:
    """A simple class to hold our NIC data and its embedding."""
    def __init__(self, code: str, description: str, embedding: list[float] = None):
        self.code = code
        self.description = description
        self.embedding = embedding

class DataService:
    """
    A service to load, index, and hold all NIC code data.
    """
    def __init__(self):
        self.nic_data: list[NicCodeRecord] = []
        self._load_and_index_data()

    def _load_and_index_data(self):
        print("Starting data loading and indexing process...")
        
        # --- MODIFIED SECTION: Read from CSV file ---
        try:
            # Construct the path to the CSV file
            file_path = os.path.join(os.path.dirname(__file__), '..', 'nic_2008_all_codes.csv')
            
            # Use pandas to read the CSV
            df = pd.read_csv(file_path)
            
            # IMPORTANT: Check your CSV file for the exact column names. 
            # You might need to change 'niccode' and 'nicdesc' below.
            # Use df.columns to see the names if you get an error.
            df = df[['niccode', 'nicdesc']] # Keep only the columns we need
            df = df.dropna() # Remove any rows with missing data
            
            # Convert the pandas DataFrame to a list of dictionaries
            raw_data = df.to_dict('records')
            print(f"Successfully loaded {len(raw_data)} records from CSV.")

        except FileNotFoundError:
            print(f"ERROR: The data file was not found. Make sure 'nic_2008_all_codes.csv' is in your project root folder.")
            return # Stop if the file doesn't exist
        
        # --- The rest of the code is the same ---

        # For each record, we generate its embedding and store it.
        # This is the "indexing" process. It runs only once at startup.
        for item in raw_data:
            embedding = inference_service.get_embedding(item["nicdesc"])
            record = NicCodeRecord(
                code=str(item["niccode"]), # Convert code to string just in case
                description=item["nicdesc"], 
                embedding=embedding
            )
            self.nic_data.append(record)
        
        print(f"Indexing complete. {len(self.nic_data)} records loaded and vectorized.")

    def get_all(self) -> list[NicCodeRecord]:
        """Returns the complete list of indexed NIC records."""
        return self.nic_data

# --- SINGLETON INSTANCE ---
data_service = DataService()