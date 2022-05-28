"""A class to help us interact with a SQL database
"""

import sqlite3 as sql

class DBInterface:
    """A basic SQLite interface

          Attributes:
              sqlite_db - Path to a SQLite DB
    """
    def __init__(self, sqlite_db: str):
        self.sqlite_db = sqlite_db

    def run_closing_query(self, query: str, params=()):
        """Public version of _run_closing_query

        Arguments:
            query    - Query to run
            params - Parameters to the query

        Returns:
            results - Results from the query
        """
        return self._run_closing_query(query, params=params)

    def _run_closing_query(self, query, params=()):
        """Public version of _run_closing_query

        Arguments:
            query    - Query to run
            params - Parameters to the query

        Returns:
            results - Results from the query
        """
        results = None
        with sql.connect(self.sqlite_db) as con:
            cur = con.cursor()
            cur.execute(query, tuple(params))
            results = cur.fetchall()
        return results

    # Getters / setters
    def get_sqlite_db(self):
        """Returns current sqlite DB
        """
        return self.sqlite_db

    def set_sqlite_db(self, sqlite_db):
        """Sets current sqlite DB
        """
        self.sqlite_db = sqlite_db
