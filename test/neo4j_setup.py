
from neo4j import GraphDatabase

# Connect with your current credentials
uri = "bolt://k2bio_neo4j:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4j"))

with driver.session(database="system") as session:
    # Replace 'new_password' with your desired new password
    query = "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'jove_llm_k2bio'"
    session.run(query)

print("Password has been changed successfully!")

