import pickle
import sqlite3
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load schemes
df = pd.read_csv(r'C:\Users\akhil\py programs\THE SIH\schemes_final_v2.csv')

# Encode for model prediction
def map_gender(val):
    return 1 if val == 'female' else 0

def map_caste(val):
    mapping = {'General': 0, 'OBC': 1, 'SC': 2, 'ST': 3}
    return mapping.get(val, 0)

def map_location(val):
    if val == 'rural': return 1
    return 0  # urban or both → 0

# Connect to DB
conn = sqlite3.connect("schemes.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS schemes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    gender TEXT,
    caste TEXT,
    income_limit INTEGER,
    location TEXT,
    age INTEGER,
    bpl_card INTEGER,
    business_type TEXT,
    link TEXT,
    state TEXT,
    marginalization_score REAL
)
""")

# Insert each scheme with predicted score
for _, row in df.iterrows():
    X = pd.DataFrame([{
        'income':     row['income'],
        'gender':     map_gender(row['gender']),
        'caste':      map_caste(row['caste']),
        'disability': row['disability'],
        'location':   map_location(row['location']),
        'bpl_card':   row['bpl_card']
    }])

    score = round(model.predict(X)[0], 2)

    cursor.execute("""
        INSERT OR REPLACE INTO schemes 
        (id, name, description, gender, caste, income_limit, 
         location, age, bpl_card, business_type, link, state, marginalization_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(row['scheme_id']),
        row['scheme_name'],
        row['scheme_description'],
        row['gender'],
        row['caste'],
        int(row['income']),
        row['location'],
        int(row['age']),
        int(row['bpl_card']),
        row['business_type'],
        row['link'],
        row['state'],
        score
    ))
    print(f"✅ {row['scheme_name']} → score: {score}")

conn.commit()
conn.close()
print("\n🎉 Database created successfully!")