import pandas as pd
import sqlite3

%pwd
#populating the database
conn = sqlite3.connect("/home/german/Desktop/insight_project/code/database/yield.sqlite")


PATH = "/home/german/Desktop/tricar/code/stack_13_17.csv"
df = pd.read_csv(PATH)

df.head()

table_name = 'produccion'

df.to_sql(table_name, conn, if_exists="replace")
df.columns
pd.read_sql_query("select * from produccion", conn)
