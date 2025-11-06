import os
import psycopg2
import pandas as pd
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

# DB config - update for your PostgreSQL instance
DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
DB_NAME = os.getenv("DB_NAME", "incentives_db")
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(__file__)))
DROP_MATCHES_TABLE = os.getenv("DROP_MATCHES_TABLE", "false").lower() == "true"
DB_RESET_ON_START = os.getenv("DB_RESET_ON_START", "false").lower() == "true"
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

def create_database_and_tables():
    """Connect to PostgreSQL, create DB/tables/indexes, enable extensions."""
    print(f"Connecting to PostgreSQL instance at {DB_CONFIG['host']}...")
    # Connect to the default 'postgres' database first to create our new DB
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG, dbname="postgres")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Handle database reset
        if DB_RESET_ON_START:
            print(f"DB_RESET_ON_START=true: Dropping database '{DB_NAME}' and all artifacts...")
            # Terminate existing connections to the database
            cursor.execute("""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = %s
                AND pid <> pg_backend_pid();
            """, (DB_NAME,))
            cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
            print(f"Database '{DB_NAME}' dropped successfully.")

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cursor.fetchone():
            print(f"Database '{DB_NAME}' not found. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
        
        cursor.close()
        conn.close()

        # --- Connect to the new database ---
        print(f"Connecting to new database '{DB_NAME}'...")
        conn = psycopg2.connect(**DB_CONFIG, dbname=DB_NAME)
        cursor = conn.cursor()

        # Enable extensions
        print("Enabling vchord and pg_trgm extensions...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE;")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

        # Create companies table
        print("Creating 'companies' table...")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            company_name TEXT,
            cae_primary_label TEXT,
            trade_description_native TEXT,
            website TEXT,
            embedding vector({EMBEDDING_DIMENSIONS})
        );
        """)

        # Create incentives table
        print("Creating 'incentives' table...")
        # Note: vector dimensions cannot be parameterized in DDL, but EMBEDDING_DIMENSIONS 
        # comes from environment variables (not user input), so this is safe
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS incentives (
            id SERIAL PRIMARY KEY,
            incentive_project_id TEXT,
            project_id TEXT,
            incentive_program TEXT,
            title TEXT,
            description TEXT,
            ai_description TEXT,
            document_urls TEXT,
            date_publication TIMESTAMP WITH TIME ZONE,
            date_start TIMESTAMP WITH TIME ZONE,
            date_end TIMESTAMP WITH TIME ZONE,
            total_budget FLOAT,
            status TEXT,
            all_data JSONB,
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE,
            eligibility_criteria JSONB,
            source_link TEXT,
            gcs_document_urls TEXT,
            embedding vector({EMBEDDING_DIMENSIONS})
        );
        """)

        # Create or drop incentive_matches table
        if DROP_MATCHES_TABLE:
            print("Dropping 'incentive_matches' table (DROP_MATCHES_TABLE=true)...")
            cursor.execute("DROP TABLE IF EXISTS incentive_matches CASCADE;")
        
        print("Creating 'incentive_matches' table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS incentive_matches (
            match_id SERIAL PRIMARY KEY,
            incentive_id INT REFERENCES incentives(id),
            company_id INT REFERENCES companies(id),
            rank INT,
            hard_criteria_score FLOAT,
            semantic_score FLOAT,
            UNIQUE(incentive_id, company_id)
        );
        """)

        # Create vector indexes for fast ANN search using vchord
        print("Creating vector indexes (vchord)...")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_companies_embedding_vchord 
        ON companies USING vchordrq (embedding vector_l2_ops);
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_incentives_embedding_vchord 
        ON incentives USING vchordrq (embedding vector_l2_ops);
        """)
        
        # Create trigram index for fuzzy company name search
        print("Creating trigram index on company_name...")
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_companies_name_trgm 
        ON companies USING gin (company_name gin_trgm_ops);
        """)
        
        # Create index on matches for faster lookup
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_matches_incentive_rank 
        ON incentive_matches (incentive_id, rank);
        """)
        
        # Load companies CSV
        cursor.execute("SELECT COUNT(*) FROM companies;")
        if cursor.fetchone()[0] == 0:
            print("Loading companies.csv...")
            companies_path = os.path.join(DATA_DIR, 'companies.csv')
            try:
                with open(companies_path, 'r', encoding='utf-8') as f:
                    cursor.copy_expert("COPY companies(company_name, cae_primary_label, trade_description_native, website) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',')", f)
                print("Companies loaded.")
            except Exception as e:
                print(f"Error loading companies.csv: {e}")
                print(f"Ensure companies.csv exists at: {companies_path}")
        else:
            print("Companies table already populated.")

        # Load incentives CSV
        cursor.execute("SELECT COUNT(*) FROM incentives;")
        if cursor.fetchone()[0] == 0:
            print("Loading incentives.csv...")
            incentives_path = os.path.join(DATA_DIR, 'incentives.csv')
            try:
                copy_sql = """
                COPY incentives(incentive_project_id, project_id, incentive_program, title, description, ai_description, document_urls, date_publication, date_start, date_end, total_budget, status, all_data, created_at, updated_at, eligibility_criteria, source_link, gcs_document_urls) 
                FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                with open(incentives_path, 'r', encoding='utf-8') as f:
                    cursor.copy_expert(copy_sql, f)
                print("Incentives loaded.")
            except Exception as e:
                print(f"Error loading incentives.csv: {e}")
                print(f"Ensure incentives.csv exists at: {incentives_path}")
        else:
            print("Incentives table already populated.")

        conn.commit()
        print("\nDatabase setup complete.")

    except psycopg2.OperationalError as e:
        print(f"FATAL: Could not connect to PostgreSQL on {DB_CONFIG['host']}:{DB_CONFIG['port']}")
        print(f"Ensure PostgreSQL is running and user '{DB_CONFIG['user']}' exists.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    create_database_and_tables()
