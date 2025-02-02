import sqlite3

class SqliteDB:
    def __init__(self, db_name='db.sqlite3', offline=True):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.offline = offline

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def create_table(self, table_name, columns):
        """
        Create a table with the specified columns.
        :param table_name: Name of the table.
        :param columns: List of column definitions (e.g., ['id INTEGER PRIMARY KEY', 'name TEXT']).
        """
        table_name = f'"{table_name}"'  # Échapper le nom de la table
        columns_str = ', '.join(columns)  # Les colonnes doivent déjà être correctement formatées
        query = f'CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})'
        try:
            self.cursor.execute(query)
            self.conn.commit()
        except sqlite3.OperationalError as e:
            raise ValueError(f"Erreur lors de la création de la table : {e}")


    def insert(self, table_name, values, columns=['name', 'vector']):
        """
        Insert a row into a table.
        :param table_name: Name of the table.
        :param values: List of values to insert.
        """
        if self.offline == False:
            return
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['?'] * len(values))
        query = f'INSERT INTO "{table_name}" ({columns_str}) VALUES ({placeholders})'
        self.cursor.execute(query, values)
        self.conn.commit()
    

    def insert_with_check(self, table_name, values, columns=[], col_name='id'):
        """
        Insert a row into a table if the key does not already exist.
        :param table_name: Name of the table.
        :param values: List of values to insert.
        """
        if self.offline == False:
            return
        if self.check_exist(values[0], col_name=col_name, table_name=table_name):
            return
        self.insert(table_name, values, columns)

    def select(self, table_name, columns=['name', 'vector'], where=None, params=None):
        """
        Retrieve data from a table.
        :param table_name: Name of the table.
        :param columns: List of columns to retrieve.
        :param where: Optional WHERE clause (e.g., "id = ?").
        :param params: Parameters for the WHERE clause.
        :return: List of rows matching the query.
        """
        columns_str = ', '.join(columns)
        where_clause = f' WHERE {where}' if where else ''
        query = f'SELECT {columns_str} FROM {table_name}{where_clause}'
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()
    
    def select_with_orderby_and_limit(self, table_name, columns=['name', 'vector'], where=None, params=None, orderby='name', limit=5):
        """
        Retrieve data from a table with order by desc.
        :param table_name: Name of the table.
        :param columns: List of columns to retrieve.
        :param where: Optional WHERE clause (e.g., "id = ?").
        :param params: Parameters for the WHERE clause.
        :return: List of rows matching the query.
        """
        columns_str = ', '.join(columns)
        where_clause = f' WHERE {where}' if where else ''
        query = f'SELECT {columns_str} FROM {table_name}{where_clause} ORDER BY {orderby} DESC LIMIT {limit}'
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()
        

    def update(self, table_name, set_values, where, params):
        """
        Update rows in a table.
        :param table_name: Name of the table.
        :param set_values: Dictionary of column-value pairs to update.
        :param where: WHERE clause for the update (e.g., "id = ?").
        :param params: Parameters for the WHERE clause.
        """
        set_clause = ', '.join([f'{column} = ?' for column in set_values.keys()])
        query = f'UPDATE {table_name} SET {set_clause} WHERE {where}'
        self.cursor.execute(query, list(set_values.values()) + (params or []))
        self.conn.commit()

    def delete(self, table_name, where, params=None):
        """
        Delete rows from a table.
        :param table_name: Name of the table.
        :param where: WHERE clause for the delete (e.g., "id = ?").
        :param params: Parameters for the WHERE clause.
        """
        query = f'DELETE FROM {table_name} WHERE {where}'
        self.cursor.execute(query, params or ())
        self.conn.commit()

    def drop_table(self, table_name):
        """
        Drop a table if it exists.
        :param table_name: Name of the table to drop.
        """
        query = f'DROP TABLE IF EXISTS {table_name}'
        self.cursor.execute(query)
        self.conn.commit()
    
    def check_exist(self, key, col_name="name", table_name='images_vectors'):
        """
        Check if a key exists in the database.
        :param key: Key to check.
        :return: True if the key exists, False otherwise.
        """
        if self.offline == False:
            return False
        query = f"SELECT 1 FROM {table_name} WHERE {col_name} = ?"
        self.cursor.execute(query, (key,))
        return self.cursor.fetchone() is not None

    def initial_db(self, table_name='images_vectors', table_name_precision='precisions'):
        """
        Initialize the database with a default table.
        :param table_name: Name of the default table to create.
        """
        self.create_table(table_name, ['name TEXT PRIMARY KEY', 'vector TEXT'])
        self.create_table(table_name_precision, ['id TEXT PRIMARY KEY', 'distance TEXT', 'color_descriptor TEXT', 'espace_color TEXT', 'nomalisation TEXT', 'shape_descriptor TEXT', 'filter TEXT', 'texture_descriptor TEXT', 'cnn_descriptor TEXT', 'p_minowski REAL', 'canal_r INTEGER', 'canal_g INTEGER', 'canal_b INTEGER', 'dim_fen INTEGER', 'interval INTEGER', 'precision REAL'])
