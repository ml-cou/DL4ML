import os

import pandas as pd
import sqlite3

def csvToDB(csv_file):
    df = pd.read_csv(csv_file)
    file_name,_=os.path.splitext(csv_file.name)
    current_directory = os.path.dirname(__file__)
    file_location=os.path.join(current_directory,f'../data/{file_name}.db')
    conn = sqlite3.connect(file_location)
    # Write the DataFrame to a SQLite table
    df.to_sql(file_name, conn, index=False, if_exists='replace')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("CSV dataset successfully converted to SQLite database.")
