import os
import psycopg2
import uvicorn
import csv
import json
import logging
import time
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MATCHES_CSV_FILE = os.getenv("MATCHES_CSV_FILE", "matches.csv")
DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.dirname(__file__)))
RUN_MATCHING_ON_START = os.getenv("RUN_MATCHING_ON_START", "false").lower() == "true"
USE_PRECOMPUTED_MATCHES = os.getenv("USE_PRECOMPUTED_MATCHES", "true").lower() == "true"
DROP_MATCHES_TABLE = os.getenv("DROP_MATCHES_TABLE", "false").lower() == "true"
MATCH_MULTIPLIER = int(os.getenv("MATCH_MULTIPLIER", "10"))
SKIP_JSON_CHAT_REQUEST = os.getenv("SKIP_JSON_CHAT_REQUEST", "false").lower() == "true"

# Set logging level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(f"Logging initialized at level: {LOG_LEVEL}")

# DB connection pool
try:
    db_pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)
except psycopg2.OperationalError as e:
    logger.error(f"FATAL: Could not connect to PostgreSQL. Error: {e}")
    exit()

@contextmanager
def get_db_connection():
    """Get connection from pool."""
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

@contextmanager
def get_db_cursor(conn):
    """Get RealDictCursor from connection."""
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        yield cursor

# OpenAI client
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
try:
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized.")
except Exception as e:
    logger.error(f"Could not initialize OpenAI client. Error: {e}")
    exit()

def extract_incentive_criteria(incentive):
    """Extract structured criteria from incentive using LLM."""
    logger.debug(f"Extracting criteria for incentive ID: {incentive.get('id')}, Title: {incentive.get('title', '')[:50]}")
    prompt = f"""
    Analyze the following Portuguese incentive data. Your task is to extract the hard eligibility criteria.
    Respond ONLY with a valid JSON object in this exact format:
    {{"entity_types": ["PME", "..."], "sector_keywords": ["Digitization", "..."]}}

    1. entity_types: List eligible entity types (e.g., "PME", "Small Mid Cap", "Municípios").
    2. sector_keywords: List key sector keywords (e.g., "Digitization", "R&D", "Tourism", "Religious").
    
    Incentive Data:
    Title: {incentive.get('title', '')}
    Description: {incentive.get('ai_description', '')}
    Eligibility: {json.dumps(incentive.get('eligibility_criteria'), ensure_ascii=False)}

    JSON Output:
    """
    
    try:
        logger.debug(f"Calling LLM for criteria extraction (SKIP_JSON={SKIP_JSON_CHAT_REQUEST})")
        if SKIP_JSON_CHAT_REQUEST:
            # Skip JSON format request, use text format directly
            logger.debug("Using text format directly (SKIP_JSON_CHAT_REQUEST enabled)")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
        else:
            # Try with json_object format first (OpenAI), fall back to text for local LLM
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
            except Exception as format_error:
                # Local LLM did not support json_object, try without response_format
                logger.warning(f"JSON object format not supported, falling back to text mode: {format_error}")
                logger.debug("Falling back to text format")
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
        
        criteria_json = response.choices[0].message.content
        logger.debug(f"LLM response received, length: {len(criteria_json)} chars")
        
        # Extract JSON from response 
        if "```json" in criteria_json:
            criteria_json = criteria_json.split("```json")[1].split("```")[0].strip()
        elif "```" in criteria_json:
            criteria_json = criteria_json.split("```")[1].split("```")[0].strip()
        
        criteria = json.loads(criteria_json)
        logger.debug(f"Extracted criteria: entity_types={len(criteria.get('entity_types', []))}, sector_keywords={len(criteria.get('sector_keywords', []))}")
        return criteria
    except Exception as e:
        logger.error(f"LLM criteria extraction failed for incentive {incentive.get('id')}: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        return {"entity_types": [], "sector_keywords": []}

def filter_companies_by_eligibility(incentive, companies, conn):
    """
    Use LLM to filter companies based on eligibility criteria.
    Returns: list of dicts with company info and eligibility assessment
    """
    logger.debug(f"Filtering {len(companies)} companies for eligibility (incentive ID: {incentive.get('id')})")
    if not companies:
        logger.debug("No companies to filter")
        return []
    
    # Prepare company data for LLM
    companies_data = []
    for c in companies:
        companies_data.append({
            "company_id": c['company_id'],
            "company_name": c['company_name'],
            "cae_primary_label": c.get('cae_primary_label', ''),
            "trade_description": c.get('trade_description_native', '')[:200]  # Limit length
        })
    
    prompt = f"""
    Given this incentive program:
    Title: {incentive.get('title', '')}
    Description: {incentive.get('ai_description', '')}
    Eligibility Criteria: {json.dumps(incentive.get('eligibility_criteria'), ensure_ascii=False)}
    
    Evaluate if each of these companies is eligible based on their business area and description.
    Focus on whether the company's sector and activities align with the incentive's requirements.
    
    Companies:
    {json.dumps(companies_data, indent=2, ensure_ascii=False)}
    
    Respond with a JSON array:
    [{{"company_id": 123, "eligible": true, "reason": "Brief reason (max 50 chars)"}}, ...]
    
    JSON Output:
    """
    
    try:
        logger.debug(f"Calling LLM for eligibility filtering ({len(companies)} companies, SKIP_JSON={SKIP_JSON_CHAT_REQUEST})")
        if SKIP_JSON_CHAT_REQUEST:
            # Skip JSON format request, use text format directly
            logger.debug("Using text format directly for eligibility filter")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
        else:
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
            except Exception as format_error:
                logger.warning(f"JSON object format not supported for eligibility filter: {format_error}")
                logger.debug("Falling back to text format")
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
        
        result_json = response.choices[0].message.content
        logger.debug(f"Eligibility LLM response received, length: {len(result_json)} chars")
        
        # Extract JSON from response
        if "```json" in result_json:
            result_json = result_json.split("```json")[1].split("```")[0].strip()
        elif "```" in result_json:
            result_json = result_json.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(result_json)
        
        # Handle both array and object with array key
        if isinstance(parsed, dict):
            parsed = parsed.get('companies', parsed.get('results', []))
        
        # Filter eligible companies and merge with original data
        eligible_ids = {item['company_id']: item['reason'] for item in parsed if item.get('eligible', False)}
        logger.debug(f"LLM parsed {len(parsed)} assessments, {len(eligible_ids)} companies marked as eligible")
        
        filtered = []
        for c in companies:
            if c['company_id'] in eligible_ids:
                c_copy = dict(c)
                c_copy['eligibility_reason'] = eligible_ids[c['company_id']]
                filtered.append(c_copy)
        
        logger.info(f"Filtered {len(filtered)} eligible companies from {len(companies)} candidates")
        logger.debug(f"Eligibility filter complete: {len(filtered)}/{len(companies)} passed")
        return filtered
        
    except Exception as e:
        logger.error(f"LLM eligibility filtering failed: {e}")
        # On error, return all companies with generic reason
        for c in companies:
            c['eligibility_reason'] = "Eligibility check unavailable"
        return companies

def sort_companies_by_preferences(companies, incentive, conn):
    """
    Sort companies by preferences: 1) has website, 2) CAE match, 3) trade description match.
    Uses LLM to score CAE and trade description relevance to incentive.
    Returns sorted list with preference scores.
    """
    logger.debug(f"Sorting {len(companies)} companies by preferences (incentive ID: {incentive.get('id')})")
    if not companies:
        logger.debug("No companies to sort")
        return []
    
    # Prepare data for LLM scoring
    companies_for_scoring = []
    for c in companies:
        companies_for_scoring.append({
            "company_id": c['company_id'],
            "cae_primary_label": c.get('cae_primary_label', ''),
            "trade_description": c.get('trade_description_native', '')[:150]
        })
    
    prompt = f"""
    Given this incentive program:
    Title: {incentive.get('title', '')}
    Description: {incentive.get('ai_description', '')}
    
    Score each company's CAE classification and trade description for relevance to the incentive.
    Provide scores from 0-10 for:
    - cae_score: How well the CAE classification matches the incentive's sector
    - trade_score: How well the trade description aligns with the incentive's objectives
    
    Companies:
    {json.dumps(companies_for_scoring, indent=2, ensure_ascii=False)}
    
    Respond with a JSON array:
    [{{"company_id": 123, "cae_score": 8, "trade_score": 7}}, ...]
    
    JSON Output:
    """
    
    try:
        logger.debug(f"Calling LLM for preference scoring ({len(companies)} companies, SKIP_JSON={SKIP_JSON_CHAT_REQUEST})")
        if SKIP_JSON_CHAT_REQUEST:
            # Skip JSON format request, use text format directly
            logger.debug("Using text format directly for preference sorting")
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
        else:
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
            except Exception as format_error:
                logger.warning(f"JSON format not supported for preference sorting: {format_error}")
                logger.debug("Falling back to text format")
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
        
        result_json = response.choices[0].message.content
        logger.debug(f"Preference scoring LLM response received, length: {len(result_json)} chars")
        
        # Extract JSON
        if "```json" in result_json:
            result_json = result_json.split("```json")[1].split("```")[0].strip()
        elif "```" in result_json:
            result_json = result_json.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(result_json)
        if isinstance(parsed, dict):
            parsed = parsed.get('companies', parsed.get('results', parsed.get('scores', [])))
        
        # Create scores map
        scores_map = {item['company_id']: (item.get('cae_score', 5), item.get('trade_score', 5)) 
                     for item in parsed}
        
    except Exception as e:
        logger.error(f"LLM preference scoring failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        # Default scores
        scores_map = {c['company_id']: (5, 5) for c in companies}
        logger.debug("Using default scores (5, 5) for all companies")
    
    logger.debug(f"Preference scores map created: {len(scores_map)} entries")
    
    # Add scores and website flag to each company
    for c in companies:
        has_website = bool(c.get('website') and c['website'].strip())
        cae_score, trade_score = scores_map.get(c['company_id'], (5, 5))
        
        # Composite score: website (20 points) + cae_score (0-10) + trade_score (0-10)
        c['has_website'] = has_website
        c['cae_score'] = cae_score
        c['trade_score'] = trade_score
        c['preference_score'] = (20 if has_website else 0) + cae_score + trade_score
        logger.debug(f"Company {c['company_id']}: website={has_website}, cae={cae_score}, trade={trade_score}, total={c['preference_score']}")
    
    # Sort by preference_score descending, then by semantic_score ascending
    sorted_companies = sorted(companies, 
                             key=lambda x: (-x['preference_score'], x.get('semantic_score', 999)))
    
    logger.info(f"Sorted {len(sorted_companies)} companies by preferences")
    logger.debug(f"Top 3 preference scores: {[c['preference_score'] for c in sorted_companies[:3]]}")
    return sorted_companies

def find_top_companies_for_incentive(incentive_id, conn, top_n=5):
    """
    Runtime matching: find top N companies for an incentive using enhanced flow:
    1. Fetch N*MATCH_MULTIPLIER candidates by embedding similarity
    2. Filter by eligibility using LLM
    3. Sort by preferences (website, CAE match, trade description)
    4. Return top N
    
    Returns: list of dicts with company_id, company_name, semantic_score, eligibility_reason, etc.
    """
    with get_db_cursor(conn) as cursor:
        # Fetch incentive
        cursor.execute("""
            SELECT id, title, ai_description, eligibility_criteria, embedding
            FROM incentives
            WHERE id = %s AND embedding IS NOT NULL
        """, (incentive_id,))
        incentive = cursor.fetchone()
        
        if not incentive:
            logger.warning(f"Incentive {incentive_id} not found or has no embedding")
            return []
        
        # Fetch top N*MATCH_MULTIPLIER candidates by embedding similarity
        candidate_limit = top_n * MATCH_MULTIPLIER
        logger.info(f"Fetching {candidate_limit} candidate companies for incentive {incentive_id}")
        logger.debug(f"Matching parameters: top_n={top_n}, multiplier={MATCH_MULTIPLIER}, candidate_limit={candidate_limit}")
        
        query_sql = """
            SELECT
                c.id AS company_id,
                c.company_name,
                c.cae_primary_label,
                c.trade_description_native,
                c.website,
                (c.embedding <-> %s::vector) AS semantic_score
            FROM
                companies c
            WHERE
                c.embedding IS NOT NULL
            ORDER BY
                semantic_score ASC
            LIMIT %s;
        """
        
        start_time = time.perf_counter()
        cursor.execute(query_sql, (incentive['embedding'], candidate_limit))
        candidates = cursor.fetchall()
        db_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Database query completed in {db_time:.2f}ms")
        
        if not candidates:
            logger.warning(f"No candidate companies found for incentive {incentive_id}")
            return []
        
        logger.info(f"Found {len(candidates)} candidates")
        logger.debug(f"Top 3 candidate semantic scores: {[c['semantic_score'] for c in candidates[:3]]}")
        
        # Filter by eligibility using LLM
        filter_start = time.perf_counter()
        eligible_companies = filter_companies_by_eligibility(incentive, candidates, conn)
        filter_time = (time.perf_counter() - filter_start) * 1000
        logger.debug(f"Eligibility filtering completed in {filter_time:.2f}ms")
        
        if not eligible_companies:
            logger.warning(f"No eligible companies after filtering for incentive {incentive_id}")
            # If no companies pass eligibility, return top N from candidates with generic reason
            for c in candidates[:top_n]:
                c['eligibility_reason'] = "No eligibility match found"
                c['has_website'] = bool(c.get('website'))
                c['preference_score'] = 0
            return candidates[:top_n]
        
        # Sort by preferences
        sort_start = time.perf_counter()
        sorted_companies = sort_companies_by_preferences(eligible_companies, incentive, conn)
        sort_time = (time.perf_counter() - sort_start) * 1000
        logger.debug(f"Preference sorting completed in {sort_time:.2f}ms")
        
        # Return top N
        top_companies = sorted_companies[:top_n]
        
        logger.info(f"Returning {len(top_companies)} top matches for incentive {incentive_id}")
        top_matches_info = [(c['company_id'], c['company_name'][:30], round(c.get('preference_score', 0), 1)) for c in top_companies]
        logger.debug(f"Top matches: {top_matches_info}")
        total_time = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Total matching time: {total_time:.2f}ms (DB: {db_time:.2f}ms, Filter: {filter_time:.2f}ms, Sort: {sort_time:.2f}ms)")
        return top_companies

def find_top_incentives_for_company(company_id, conn, top_n=5):
    """
    Runtime matching: find top N incentives for a company.
    Returns: list of dicts with incentive_id, title, semantic_score
    """
    with get_db_cursor(conn) as cursor:
        # Fetch company
        cursor.execute("""
            SELECT id, company_name, embedding
            FROM companies
            WHERE id = %s AND embedding IS NOT NULL
        """, (company_id,))
        company = cursor.fetchone()
        
        if not company:
            logger.warning(f"Company {company_id} not found or has no embedding")
            return []
        
        # Vector search against incentives
        cursor.execute("""
            SELECT
                i.id AS incentive_id,
                i.title,
                i.ai_description,
                (i.embedding <-> %(company_embedding)s::vector) AS semantic_score
            FROM
                incentives i
            WHERE
                i.embedding IS NOT NULL
            ORDER BY
                semantic_score ASC
            LIMIT %(top_n)s;
        """, {
            "company_embedding": company['embedding'],
            "top_n": top_n
        })
        return cursor.fetchall()

def run_company_matching():
    """Match companies to incentives, save to DB and CSV."""
    logger.info("--- Starting Company Matching ---")
    
    if DROP_MATCHES_TABLE:
        logger.info("Dropping and recreating incentive_matches table...")
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cursor:
                cursor.execute("DROP TABLE IF EXISTS incentive_matches CASCADE;")
                cursor.execute("""
                    CREATE TABLE incentive_matches (
                        match_id SERIAL PRIMARY KEY,
                        incentive_id INT REFERENCES incentives(id),
                        company_id INT REFERENCES companies(id),
                        rank INT,
                        hard_criteria_score FLOAT,
                        semantic_score FLOAT,
                        UNIQUE(incentive_id, company_id)
                    );
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_matches_incentive_rank 
                    ON incentive_matches (incentive_id, rank);
                """)
                conn.commit()
    
    all_matches_for_csv = []
    
    with get_db_connection() as conn:
        with get_db_cursor(conn) as cursor:
            # Fetch incentives that haven't been matched yet
            cursor.execute("""
                SELECT i.id, i.title
                FROM incentives i
                LEFT JOIN incentive_matches m ON i.id = m.incentive_id
                WHERE m.match_id IS NULL AND i.embedding IS NOT NULL;
            """)
            
            incentives = cursor.fetchall()
            logger.info(f"Found {len(incentives)} incentives to match.")
            
            matches_to_insert = []
            
            for i, incentive in enumerate(incentives):
                logger.info(f"Matching {i+1}/{len(incentives)}: {incentive['title'][:50]}...")
                
                # Use extracted function
                start_time = time.perf_counter()
                top_companies = find_top_companies_for_incentive(incentive['id'], conn, top_n=5)
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"  Runtime matching took {elapsed:.2f}ms")
                
                # Store results
                for rank, company in enumerate(top_companies, 1):
                    match_data = (
                        incentive['id'],
                        company['company_id'],
                        rank,
                        company.get('preference_score', 0),
                        company['semantic_score']
                    )
                    matches_to_insert.append(match_data)
                    
                    all_matches_for_csv.append({
                        "incentive_id": incentive['id'],
                        "incentive_title": incentive['title'],
                        "company_id": company['company_id'],
                        "company_name": company['company_name'],
                        "rank": rank,
                        "preference_score": company.get('preference_score', 0),
                        "semantic_score": company['semantic_score'],
                        "eligibility_reason": company.get('eligibility_reason', ''),
                        "has_website": company.get('has_website', False)
                    })

            # Batch insert matches
            if matches_to_insert:
                logger.info(f"Saving {len(matches_to_insert)} matches to database...")
                execute_batch(
                    conn.cursor(),
                    """
                    INSERT INTO incentive_matches (incentive_id, company_id, rank, hard_criteria_score, semantic_score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (incentive_id, company_id) DO NOTHING;
                    """,
                    matches_to_insert
                )
                conn.commit()
    
    # Write CSV
    if all_matches_for_csv:
        csv_path = os.path.join(DATA_DIR, MATCHES_CSV_FILE)
        logger.info(f"Writing {len(all_matches_for_csv)} matches to {csv_path}...")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_matches_for_csv[0].keys())
            writer.writeheader()
            writer.writerows(all_matches_for_csv)
    
    logger.info("--- Company Matching Finished ---")

def generate_matches_csv(use_precomputed=True, batch_size=10):
    """
    Generate matches.csv from precomputed table or runtime matching.
    
    Args:
        use_precomputed: Use incentive_matches table (True) or runtime matching (False)
        batch_size: Process N incentives at a time for runtime mode
    """
    csv_path = os.path.join(DATA_DIR, MATCHES_CSV_FILE)
    logger.info(f"Generating {csv_path} (mode: {'precomputed' if use_precomputed else 'runtime'})...")
    
    all_matches = []
    
    with get_db_connection() as conn:
        with get_db_cursor(conn) as cursor:
            if use_precomputed:
                # Export from incentive_matches table
                cursor.execute("""
                    SELECT 
                        m.incentive_id,
                        i.title AS incentive_title,
                        m.company_id,
                        c.company_name,
                        m.rank,
                        m.hard_criteria_score,
                        m.semantic_score
                    FROM incentive_matches m
                    JOIN incentives i ON m.incentive_id = i.id
                    JOIN companies c ON m.company_id = c.id
                    ORDER BY m.incentive_id, m.rank;
                """)
                all_matches = cursor.fetchall()
                logger.info(f"Exported {len(all_matches)} precomputed matches")
            else:
                # Runtime matching for all incentives
                cursor.execute("""
                    SELECT id, title
                    FROM incentives
                    WHERE embedding IS NOT NULL
                    ORDER BY id;
                """)
                incentives = cursor.fetchall()
                logger.info(f"Running runtime matching for {len(incentives)} incentives...")
                
                for i, incentive in enumerate(incentives):
                    if i % batch_size == 0:
                        logger.info(f"  Processing batch {i//batch_size + 1} ({i}/{len(incentives)})...")
                    
                    start_time = time.perf_counter()
                    top_companies = find_top_companies_for_incentive(incentive['id'], conn, top_n=5)
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    for rank, company in enumerate(top_companies, 1):
                        all_matches.append({
                            "incentive_id": incentive['id'],
                            "incentive_title": incentive['title'],
                            "company_id": company['company_id'],
                            "company_name": company['company_name'],
                            "rank": rank,
                            "preference_score": company.get('preference_score', 0),
                            "semantic_score": company['semantic_score'],
                            "eligibility_reason": company.get('eligibility_reason', ''),
                            "has_website": company.get('has_website', False),
                            "matching_time_ms": elapsed if rank == 1 else None
                        })
                
                logger.info(f"Generated {len(all_matches)} runtime matches")
    
    # Write CSV
    if all_matches:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Convert dict rows if from precomputed (RealDictRow objects)
            if use_precomputed and all_matches:
                all_matches = [dict(row) for row in all_matches]
            
            writer = csv.DictWriter(f, fieldnames=all_matches[0].keys())
            writer.writeheader()
            writer.writerows(all_matches)
        logger.info(f"CSV written to {csv_path}")
    else:
        logger.warning("No matches to write to CSV")

app = FastAPI(
    title="Incentives AI Chatbot",
    description="Match companies with public incentives."
)

# Enable CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    query: str
    user_id: str | None = None

class MatchRequest(BaseModel):
    incentive_id: int

class CompanyMatch(BaseModel):
    company_id: int
    company_name: str
    rank: int
    eligibility_reason: str
    has_website: bool
    semantic_score: float
    preference_score: float

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate embedding for text."""
    logger.debug(f"Generating embedding (model: {model}, text length: {len(text)} chars)")
    embed_start = time.perf_counter()
    response = client.embeddings.create(input=[text], model=model)
    embed_time = (time.perf_counter() - embed_start) * 1000
    embedding = response.data[0].embedding
    logger.debug(f"Embedding generated in {embed_time:.2f}ms, dimension: {len(embedding)}")
    return embedding

async def classify_intent(query: str):
    """Classify user intent with LLM."""
    logger.debug(f"Classifying intent for query: {query[:100]}")
    prompt = f"""
    Classify the user's query into ONE of these categories:
    - 'company_specific': Asking about a specific company by name
    - 'company_general': General question about companies or searching for companies
    - 'incentive_specific': Asking about a specific incentive program
    - 'incentive_general': General question about incentives or searching for incentives
    - 'match_search': Asking which companies fit a specific incentive
    - 'other': Greetings, unclear questions, or off-topic
    
    Respond with only the category name.
    
    Examples:
    - "Tell me about DANIJO - INDÚSTRIA DE CONFECÇÃO, LDA": company_specific
    - "What companies work in textiles?": company_general
    - "Details on the SIID program": incentive_specific
    - "What incentives exist for SMEs?": incentive_general
    - "Which companies are good for SIID?": match_search
    - "Hello": other

    Query: "{query}"
    Classification:
    """
    try:
        logger.debug(f"Calling LLM for intent classification (model: {CHAT_MODEL})")
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        intent = response.choices[0].message.content.strip().lower()
        logger.debug(f"LLM raw intent response: '{intent}'")
        valid_intents = ['company_specific', 'company_general', 'incentive_specific', 
                        'incentive_general', 'match_search', 'other']
        if intent not in valid_intents:
            logger.debug(f"Invalid intent '{intent}', defaulting to 'other'")
            return 'other'
        logger.debug(f"Classified intent: {intent}")
        return intent
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        return 'other'

async def retrieve_context(query: str, intent: str):
    """Retrieve context from DB based on intent. Returns (context, metrics)."""
    logger.debug(f"Retrieving context for intent: {intent}, query: {query[:100]}")
    context = ""
    metrics = {"matching_mode": None, "matching_time_ms": None, "match_direction": None}
    
    with get_db_connection() as conn:
        with get_db_cursor(conn) as cursor:
            if intent == 'company_specific':
                # Direct company name search
                logger.debug(f"Executing company_specific search with query: {query}")
                cursor.execute(
                    """
                    SELECT company_name, trade_description_native, cae_primary_label, website
                    FROM companies
                    WHERE company_name ILIKE %s
                    LIMIT 3;
                    """,
                    (f"%{query}%",)
                )
                results = cursor.fetchall()
                context = json.dumps([dict(r) for r in results], indent=2, ensure_ascii=False)
            
            elif intent == 'company_general':
                # General company search with clarification
                logger.debug("Generating embedding for company_general search")
                query_embedding = get_embedding(query)
                logger.debug(f"Executing company_general vector search")
                cursor.execute(
                    """
                    SELECT company_name, trade_description_native, cae_primary_label
                    FROM companies
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                context = json.dumps({
                    "needs_clarification": True,
                    "results": [dict(r) for r in results]
                }, indent=2, ensure_ascii=False)
            
            elif intent == 'incentive_specific':
                # Direct incentive search
                logger.debug("Generating embedding for incentive_specific search")
                query_embedding = get_embedding(query)
                logger.debug(f"Executing incentive_specific vector search")
                cursor.execute(
                    """
                    SELECT title, ai_description, eligibility_criteria::text
                    FROM incentives
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s::vector
                    LIMIT 1;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                context = json.dumps([dict(r) for r in results], indent=2, ensure_ascii=False)
            
            elif intent == 'incentive_general':
                # General incentive search with clarification
                logger.debug("Generating embedding for incentive_general search")
                query_embedding = get_embedding(query)
                logger.debug(f"Executing incentive_general vector search (top 5)")
                cursor.execute(
                    """
                    SELECT title, ai_description, eligibility_criteria::text
                    FROM incentives
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s::vector
                    LIMIT 5;
                    """,
                    (query_embedding,)
                )
                results = cursor.fetchall()
                context = json.dumps({
                    "needs_clarification": True,
                    "results": [dict(r) for r in results]
                }, indent=2, ensure_ascii=False)
            
            elif intent == 'match_search':
                # Find incentive by semantic search
                logger.debug("Generating embedding for match_search")
                query_embedding = get_embedding(query)
                logger.debug(f"Executing match_search vector search")
                cursor.execute(
                    """
                    SELECT id, title FROM incentives
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <-> %s::vector
                    LIMIT 1;
                    """,
                    (query_embedding,)
                )
                incentive = cursor.fetchone()
                if not incentive:
                    logger.debug("No matching incentive found for match_search")
                    return "No matching incentive found.", metrics
                logger.debug(f"Found incentive: ID={incentive['id']}, Title={incentive['title'][:50]}")
                
                metrics["match_direction"] = "incentive_to_companies"
                metrics["incentive_id"] = incentive['id']
                
                if USE_PRECOMPUTED_MATCHES:
                    # Get pre-computed matches from table
                    logger.debug(f"Using precomputed matches for incentive {incentive['id']}")
                    start_time = time.perf_counter()
                    cursor.execute(
                        """
                        SELECT c.company_name, c.id as company_id, m.hard_criteria_score, m.semantic_score
                        FROM incentive_matches m
                        JOIN companies c ON m.company_id = c.id
                        WHERE m.incentive_id = %s
                        ORDER BY m.rank ASC
                        LIMIT 5;
                        """,
                        (incentive['id'],)
                    )
                    results = cursor.fetchall()
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    metrics["matching_mode"] = "precomputed"
                    metrics["matching_time_ms"] = elapsed
                    logger.info(f"Precomputed match retrieval took {elapsed:.2f}ms")
                else:
                    # Runtime matching
                    logger.debug(f"Using runtime matching for incentive {incentive['id']}")
                    start_time = time.perf_counter()
                    results = find_top_companies_for_incentive(incentive['id'], conn, top_n=5)
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    metrics["matching_mode"] = "runtime"
                    metrics["matching_time_ms"] = elapsed
                    logger.info(f"Runtime matching (incentive->companies) took {elapsed:.2f}ms")
                    logger.debug(f"Runtime matching returned {len(results)} companies")
                
                context = json.dumps({
                    "incentive_matched": incentive['title'],
                    "incentive_id": incentive['id'],
                    "top_5_companies": [dict(r) for r in results]
                }, indent=2, ensure_ascii=False)
            
            else:
                context = "General conversation."
                
    return context, metrics

async def synthesize_answer(query: str, context: str):
    """Generate natural language answer from context."""
    logger.debug(f"Synthesizing answer for query: {query[:100]}")
    logger.debug(f"Context length: {len(context)} chars, first 200 chars: {context[:200]}")
    # Check if context requires clarification
    try:
        context_obj = json.loads(context) if context.startswith('{') or context.startswith('[') else None
        if context_obj and isinstance(context_obj, dict) and context_obj.get('needs_clarification'):
            prompt = f"""
            You are a helpful assistant specialized in Portuguese public incentives.
            The user asked a general question and we found multiple matching results.
            Ask the user (in Portuguese) if any of the results match what they're looking for.
            Present the options clearly and ask them to specify which one they want.

            User Question:
            {query}

            Results Found:
            {json.dumps(context_obj.get('results', []), indent=2, ensure_ascii=False)}

            Respond in Portuguese:
            """
        else:
            prompt = f"""
            You are a helpful assistant specialized in Portuguese public incentives.
            Answer the user's question based ONLY on the provided context.
            If the context is empty or not relevant, say you couldn't find information.
            Respond in Portuguese.

            Context:
            {context}

            User Question:
            {query}

            Answer:
            """
    except:
        # If context is not JSON, use standard prompt
        prompt = f"""
        You are a helpful assistant specialized in Portuguese public incentives.
        Answer the user's question based ONLY on the provided context.
        If the context is empty or not relevant, say you couldn't find information.
        Respond in Portuguese.

        Context:
        {context}

        User Question:
        {query}

        Answer:
        """
    
    try:
        logger.debug(f"Calling LLM for answer synthesis (model: {CHAT_MODEL}, temp: 0.2)")
        synthesis_start = time.perf_counter()
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content
        synthesis_time = (time.perf_counter() - synthesis_start) * 1000
        logger.debug(f"Answer synthesis completed in {synthesis_time:.2f}ms, answer length: {len(answer)} chars")
        return answer
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="LLM synthesis failed.")

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    """RAG pipeline: classify -> retrieve -> synthesize."""
    logger.info(f"Chat endpoint called: query='{query.query[:100]}', user_id={query.user_id}")
    start_time = time.perf_counter()
    
    intent = await classify_intent(query.query)
    logger.debug(f"Intent classified: {intent}")
    context, metrics = await retrieve_context(query.query, intent)
    logger.debug(f"Context retrieved: length={len(context)} chars, metrics={metrics}")
    answer = await synthesize_answer(query.query, context)
    
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    logger.info(f"Chat endpoint completed in {total_time:.2f}ms (intent: {intent})")
    
    return {
        "answer": answer,
        "intent": intent,
        "context_retrieved": context,
        "total_time_ms": (end_time - start_time) * 1000,
        "matching_mode": metrics["matching_mode"],
        "matching_time_ms": metrics["matching_time_ms"],
        "match_direction": metrics["match_direction"],
        "incentive_id": metrics.get("incentive_id")  # Include incentive_id when available for CSV download
    }

@app.post("/chat/stream")
async def chat_stream_endpoint(query: ChatQuery):
    """Streaming RAG pipeline for faster first token."""
    logger.info(f"Chat stream endpoint called: query='{query.query[:100]}', user_id={query.user_id}")
    intent = await classify_intent(query.query)
    logger.debug(f"Intent classified: {intent}")
    context, metrics = await retrieve_context(query.query, intent)
    logger.debug(f"Context retrieved: length={len(context)} chars")
    
    # Check if context requires clarification
    try:
        context_obj = json.loads(context) if context.startswith('{') or context.startswith('[') else None
        if context_obj and isinstance(context_obj, dict) and context_obj.get('needs_clarification'):
            prompt = f"""
            You are a helpful assistant specialized in Portuguese public incentives.
            The user asked a general question and we found multiple matching results.
            Ask the user (in Portuguese) if any of the results match what they're looking for.
            Present the options clearly and ask them to specify which one they want.

            User Question:
            {query.query}

            Results Found:
            {json.dumps(context_obj.get('results', []), indent=2, ensure_ascii=False)}

            Respond in Portuguese:
            """
        else:
            prompt = f"""
            You are a helpful assistant specialized in Portuguese public incentives.
            Answer the user's question based ONLY on the provided context.
            If the context is empty or not relevant, say you couldn't find information.
            Respond in Portuguese.

            Context:
            {context}

            User Question:
            {query.query}

            Answer:
            """
    except:
        prompt = f"""
        You are a helpful assistant specialized in Portuguese public incentives.
        Answer the user's question based ONLY on the provided context.
        If the context is empty or not relevant, say you couldn't find information.
        Respond in Portuguese.

        Context:
        {context}

        User Question:
        {query.query}

        Answer:
        """
    
    def generate():
        try:
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.2
            )
            yield "event: start\ndata: {}\n\n"
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/")
def root():
    return {"message": "Incentives AI API. POST to /chat or /chat/stream"}

@app.post("/matches")
async def get_matches(request: MatchRequest):
    """
    Get top 5 company matches for a specific incentive as JSON.
    Performs runtime matching using the enhanced LLM-based filtering and preference sorting.
    """
    logger.info(f"POST /matches endpoint called for incentive_id: {request.incentive_id}")
    try:
        with get_db_connection() as conn:
            start_time = time.perf_counter()
            top_companies = find_top_companies_for_incentive(request.incentive_id, conn, top_n=5)
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Match retrieval completed in {elapsed:.2f}ms, found {len(top_companies)} companies")
            
            if not top_companies:
                raise HTTPException(status_code=404, detail=f"No matches found for incentive {request.incentive_id}")
            
            # Convert to response format
            matches = []
            for rank, company in enumerate(top_companies, 1):
                matches.append({
                    "company_id": company['company_id'],
                    "company_name": company['company_name'],
                    "rank": rank,
                    "eligibility_reason": company.get('eligibility_reason', ''),
                    "has_website": company.get('has_website', False),
                    "semantic_score": float(company['semantic_score']),
                    "preference_score": company.get('preference_score', 0)
                })
            
            return {
                "incentive_id": request.incentive_id,
                "matches": matches,
                "matching_time_ms": elapsed
            }
    except Exception as e:
        logger.error(f"Error getting matches for incentive {request.incentive_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/matches/csv")
async def download_matches_csv(incentive_id: int = None):
    """
    Download matches as CSV.
    - If incentive_id is provided: returns matches for that specific incentive
    - If incentive_id is None: returns all matches from the database or generates full CSV
    """
    import io
    from fastapi.responses import Response
    
    logger.info(f"GET /matches/csv endpoint called with incentive_id: {incentive_id}")
    try:
        output = io.StringIO()
        
        with get_db_connection() as conn:
            with get_db_cursor(conn) as cursor:
                if incentive_id is not None:
                    logger.debug(f"Generating CSV for specific incentive: {incentive_id}")
                    # Get matches for specific incentive
                    if USE_PRECOMPUTED_MATCHES:
                        # Try precomputed first
                        cursor.execute("""
                            SELECT 
                                m.incentive_id,
                                i.title AS incentive_title,
                                m.company_id,
                                c.company_name,
                                m.rank,
                                m.hard_criteria_score AS preference_score,
                                m.semantic_score
                            FROM incentive_matches m
                            JOIN incentives i ON m.incentive_id = i.id
                            JOIN companies c ON m.company_id = c.id
                            WHERE m.incentive_id = %s
                            ORDER BY m.rank;
                        """, (incentive_id,))
                        matches = cursor.fetchall()
                        
                        if not matches:
                            # Fall back to runtime matching
                            logger.info(f"No precomputed matches for incentive {incentive_id}, using runtime matching")
                            top_companies = find_top_companies_for_incentive(incentive_id, conn, top_n=5)
                            
                            # Get incentive title
                            cursor.execute("SELECT title FROM incentives WHERE id = %s", (incentive_id,))
                            incentive_row = cursor.fetchone()
                            incentive_title = incentive_row['title'] if incentive_row else f"Incentive {incentive_id}"
                            
                            matches = []
                            for rank, company in enumerate(top_companies, 1):
                                matches.append({
                                    "incentive_id": incentive_id,
                                    "incentive_title": incentive_title,
                                    "company_id": company['company_id'],
                                    "company_name": company['company_name'],
                                    "rank": rank,
                                    "preference_score": company.get('preference_score', 0),
                                    "semantic_score": company['semantic_score'],
                                    "eligibility_reason": company.get('eligibility_reason', ''),
                                    "has_website": company.get('has_website', False)
                                })
                    else:
                        # Runtime matching
                        top_companies = find_top_companies_for_incentive(incentive_id, conn, top_n=5)
                        
                        # Get incentive title
                        cursor.execute("SELECT title FROM incentives WHERE id = %s", (incentive_id,))
                        incentive_row = cursor.fetchone()
                        incentive_title = incentive_row['title'] if incentive_row else f"Incentive {incentive_id}"
                        
                        matches = []
                        for rank, company in enumerate(top_companies, 1):
                            matches.append({
                                "incentive_id": incentive_id,
                                "incentive_title": incentive_title,
                                "company_id": company['company_id'],
                                "company_name": company['company_name'],
                                "rank": rank,
                                "preference_score": company.get('preference_score', 0),
                                "semantic_score": company['semantic_score'],
                                "eligibility_reason": company.get('eligibility_reason', ''),
                                "has_website": company.get('has_website', False)
                            })
                else:
                    # Get all matches from database
                    logger.debug("Generating CSV for all matches")
                    cursor.execute("""
                        SELECT 
                            m.incentive_id,
                            i.title AS incentive_title,
                            m.company_id,
                            c.company_name,
                            m.rank,
                            m.hard_criteria_score AS preference_score,
                            m.semantic_score
                        FROM incentive_matches m
                        JOIN incentives i ON m.incentive_id = i.id
                        JOIN companies c ON m.company_id = c.id
                        ORDER BY m.incentive_id, m.rank;
                    """)
                    matches = cursor.fetchall()
                
                if not matches:
                    raise HTTPException(status_code=404, detail="No matches found")
                
                # Convert to dict if needed
                matches = [dict(m) for m in matches]
                
                # Write CSV
                if matches:
                    writer = csv.DictWriter(output, fieldnames=matches[0].keys())
                    writer.writeheader()
                    writer.writerows(matches)
        
        # Return CSV response
        csv_content = output.getvalue()
        filename = f"matches_incentive_{incentive_id}.csv" if incentive_id else "matches_all.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main execution
if __name__ == "__main__":
    # Run matching if env var set
    if RUN_MATCHING_ON_START:
        run_company_matching()
    
    # Start API server
    logger.info("Starting FastAPI server...")
    logger.info("Docs at http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)