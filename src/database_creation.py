#Creation of Database
import sqlite3

def create_legal_database():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect("legal_assistant.db")
    cursor = conn.cursor()
    
    # Create table for storing legal articles
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS legal_articles (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT,
        keywords TEXT
    )
    """)
    
    # Insert example articles (include complete articles here)
    cursor.execute("INSERT INTO legal_articles (title, content, keywords) VALUES (?, ?, ?)", (
        "Legal Contracts", 
        """A legal contract is a binding agreement between two or more parties that creates mutual obligations enforceable by law. Key elements include:
        - Offer
        - Acceptance
        - Consideration
        - Capacity
        - Legality
        Contracts can be oral or written, but certain contracts must be in writing to be enforceable (e.g., real estate transactions).""",
        "contract, agreement, legal, binding, legal contract"
    ))
    cursor.execute("INSERT INTO legal_articles (title, content, keywords) VALUES (?, ?, ?)", (
        "Lawyer Requirements", 
        """To become a lawyer, you typically need to:
        1. Complete a law degree (e.g., JD).
        2. Pass the bar examination.
        3. Meet character and fitness requirements.
        4. Complete continuing education for license renewal.
        Licensing requirements vary by jurisdiction.""",
        "lawyer, requirements, bar, degree"
    ))
    
    
    # Commit and close
    conn.commit()
    # Check if the table exists
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(cursor.fetchall())

    # Check data in the table
    #cursor.execute("SELECT * FROM legal_articles;")
    #rows = cursor.fetchall()
    #print(rows)  # Verify the rows in the database

    conn.close()

# Initialize database (run this once)
create_legal_database()