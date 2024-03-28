import streamlit as st
import mysql.connector


def init_connection():
    return mysql.connector.connect(**st.secrets["mysql"])


conn = init_connection()


@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
