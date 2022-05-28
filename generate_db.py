"""Helper script to generate a database with users
and hashed passwords
"""
import json
import sqlite3 as sql
import os
import bcrypt

DATA_PATH = os.environ["DATA_PATH"]
DB_SALT   = os.environ["DATABASE_SALT"].encode('utf-8')
DB_PATH   = os.path.join(DATA_PATH, "database", "users.db")

# remove the old DB in favor of our newly seeded one
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# Creating a basic table for us to store users in
BASE_STATEMENT = """
CREATE TABLE IF NOT EXISTS user (
  name varchar(50) DEFAULT NULL,
  password varchar(100) DEFAULT NULL
);
"""
USER_STATEMENT = "INSERT INTO user (name, password) VALUES"

with open(os.path.join(DATA_PATH, "users", "passwords.json"), "r", encoding="utf-8") as config:
    json_config = json.loads(config.read())

for user in json_config["passwords"]:
    # Hash to password using Bcrypt
    hashed = bcrypt.hashpw(
        user['pass'].encode('utf-8'),
        DB_SALT
    ).decode('utf-8')
    USER_STATEMENT += f"\n\t('{user['name']}', '{hashed}')"

USER_STATEMENT += ";"

# Run the SQL on our database
with sql.connect(DB_PATH) as con:
    cur = con.cursor()
    cur.execute(BASE_STATEMENT)

with sql.connect(DB_PATH) as con:
    cur = con.cursor()
    cur.execute(USER_STATEMENT)
