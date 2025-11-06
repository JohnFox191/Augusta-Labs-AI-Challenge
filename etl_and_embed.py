import os
import psycopg2
import time
from openai import OpenAI
import numpy as np
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()

# Config
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "incentives_db")
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# Initialize OpenAI client
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
try:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Set OPENAI_API_KEY environment variable.")
    exit()

def get_db_connection():
    """Establish DB connection."""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def get_embeddings_batch(texts, model=EMBEDDING_MODEL):
    """Generate embeddings for a batch of texts."""
    if not texts:
        return []
    try:
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return [None] * len(texts)

def process_batch(cursor, table_name, id_field, text_fields):
    """Fetch batch, generate embeddings, update DB."""
    print(f"Fetching batch of {BATCH_SIZE} from '{table_name}'...")
    
    document_sql = " || ' | ' || ".join(text_fields)
    
    cursor.execute(f"""
        SELECT {id_field}, {document_sql}
        FROM {table_name}
        WHERE embedding IS NULL
        LIMIT {BATCH_SIZE};
    """)
    
    records = cursor.fetchall()
    if not records:
        print(f"No more records to embed in '{table_name}'.")
        return 0

    print(f"Generating {len(records)} embeddings...")
    texts = [text for _, text in records]
    embeddings = get_embeddings_batch(texts)
    
    embeddings_to_update = [
        (embeddings[i], records[i][0]) 
        for i in range(len(records)) 
        if embeddings[i] is not None
    ]

    if embeddings_to_update:
        print(f"Updating {len(embeddings_to_update)} embeddings...")
        execute_batch(
            cursor,
            f"UPDATE {table_name} SET embedding = %s WHERE {id_field} = %s",
            embeddings_to_update
        )
    
    return len(records)

def main():
    """Generate embeddings for companies and incentives."""
    print("Starting embedding generation...")
    
    print("\n--- Processing Companies ---")
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cursor:
                total_processed = 0
                while True:
                    processed = process_batch(
                        cursor, 
                        table_name="companies", 
                        id_field="id", 
                        text_fields=["COALESCE(company_name, '')", "COALESCE(cae_primary_label, '')", "COALESCE(trade_description_native, '')"]
                    )
                    total_processed += processed
                    if processed < BATCH_SIZE:
                        break
                print(f"Companies done. Total: {total_processed}")
    except Exception as e:
        print(f"Error processing companies: {e}")
    finally:
        conn.close()

    # --- Process Incentives ---
    print("\n--- Processing Incentives ---")
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cursor:
                total_processed = 0
                while True:
                    processed = process_batch(
                        cursor,
                        table_name="incentives",
                        id_field="id",
                        text_fields=["COALESCE(title, '')", "COALESCE(ai_description, '')", "COALESCE(eligibility_criteria::text, '')"]
                    )
                    total_processed += processed
                    if processed < BATCH_SIZE:
                        break
                print(f"Incentives done. Total: {total_processed}")
    except Exception as e:
        print(f"Error processing incentives: {e}")
    finally:
        conn.close()

    print("\nEmbedding generation complete.")

if __name__ == "__main__":
    main()