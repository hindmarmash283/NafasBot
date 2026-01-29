import sqlite3
import hashlib
from datetime import datetime, timedelta
import pandas as pd

def init_database():
    conn = sqlite3.connect('nafasbot.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id))''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
        msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id))''')
   
    conn.commit()
    return conn


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(conn, username, password):
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                      (username, hash_password(password)))
        conn.commit()
        return True, "تم إنشاء الحساب بنجاح!"
    except sqlite3.IntegrityError:
        return False, "اسم المستخدم موجود مسبقاً"

def login_user(conn, username, password):
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, username FROM users WHERE username = ? AND password_hash = ?', 
                  (username, hash_password(password)))
    return cursor.fetchone()

def create_new_session(conn, user_id, first_msg):
    title = first_msg[:30] + "..."
    cur = conn.execute('INSERT INTO sessions (user_id, title) VALUES (?, ?)', (user_id, title))
    conn.commit()
    return cur.lastrowid

def get_user_sessions(conn, user_id):
    cur = conn.execute('SELECT session_id, title, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC', (user_id,))
    return cur.fetchall()

def delete_session(conn, session_id):
    conn.execute('DELETE FROM messages WHERE session_id=?', (session_id,))
    conn.execute('DELETE FROM sessions WHERE session_id=?', (session_id,))
    conn.commit()

def rename_session(conn, session_id, new_title):
    conn.execute('UPDATE sessions SET title=? WHERE session_id=?', (new_title, session_id))
    conn.commit()

def save_message(conn, session_id, user_id, question, answer, category):
    if not category: category = "General"
    conn.execute('''
            INSERT INTO messages (session_id, user_id, question, answer, category) 
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_id, question, answer, category))
    conn.commit()

def get_session_messages(conn, session_id):
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM messages WHERE session_id=? ORDER BY msg_id ASC', (session_id,))
    return cursor.fetchall()

def get_new_training_data(conn):
    # نسحب الأسئلة والأجوبة والتصنيف من آخر أسبوع
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    query = "SELECT question, answer, category FROM messages WHERE timestamp > ?"
    df = pd.read_sql_query(query, conn, params=(one_week_ago,))
    
    # تنظيف البيانات وتجهيزها للدمج
    df = df.rename(columns={'question': 'Question', 'answer': 'Answer', 'category': 'Hierarchical Diagnosis'})
    return df
