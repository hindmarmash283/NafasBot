import streamlit as st
import sqlite3
import hashlib
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os
import google.generativeai as genai
import re
import pyarabic.araby as araby
from nltk.stem.isri import ISRIStemmer
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# ==========================================
# 2. قسم قاعدة البيانات (Database Manager)
# ==========================================
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
    one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    query = "SELECT question, answer, category FROM messages WHERE timestamp > ?"
    df = pd.read_sql_query(query, conn, params=(one_week_ago,))
    df = df.rename(columns={'question': 'Question', 'answer': 'Answer', 'category': 'Hierarchical Diagnosis'})
    return df

# ==========================================
# 3. قسم محرك الذكاء (AI Engine)
# ==========================================
def normalize_arabic_word(word):
    word = araby.strip_tatweel(word)
    word = araby.strip_tashkeel(word)
    word = re.sub(r'[إأآا]', 'ا', word)
    word = re.sub(r'ى', 'ي', word)
    word = re.sub(r'ؤ', 'ء', word)
    word = re.sub(r'ئ', 'ء', word)
    word = re.sub(r'ة', 'ه', word)
    word = re.sub(r'(.)\1{2,}', r'\1', word)
    return word

def stem_arabic_word(text):
    stemmer = ISRIStemmer()
    try:
        text = normalize_arabic_word(text)
        words = text.split()
        return " ".join([stemmer.stem(word) for word in words])
    except: return text

@st.cache_resource
def load_nafsbot_models():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBZoUp0GODMNe6tCmdwOEAF7GBjc2Pmsdw" 
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    try:
        svm_model, df_data = None, None
        if os.path.exists('svm_model.zip'):
            with zipfile.ZipFile('svm_model.zip', 'r') as z:
                pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f: svm_model = pickle.load(f)
        elif os.path.exists('svm_model.pkl'):
            with open('svm_model.pkl', 'rb') as f: svm_model = pickle.load(f)

        if os.path.exists('dataset_original.zip'):
            with zipfile.ZipFile('dataset_original.zip', 'r') as z:
                pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f: df_data = pd.read_pickle(f)
        elif os.path.exists('dataset_original.pkl'):
            df_data = pd.read_pickle('dataset_original.pkl')

        if os.path.exists('vectorizer.pkl'):
            with open('vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f: enc = pickle.load(f)
        
        if svm_model is None or df_data is None: return None
        return {'model': model, 'svm': svm_model, 'vectorizer': vec, 
                'encoder': enc, 'data': df_data, 'stem': stem_arabic_word}
    except: return None

def get_nafsbot_response(models, patient_input, chat_history):
    try:
        processed = models['stem'](patient_input)
        vec = models['vectorizer'].transform([processed]).toarray()
        pred_idx = models['svm'].predict(vec)[0]
        category = models['encoder'].inverse_transform([pred_idx])[0]
        
        related = models['data'][models['data']['Hierarchical Diagnosis'] == category]
        if len(related) == 0:
            return category, "ما عندي معلومة طبية دقيقة عن حالتك في قاعدة بياناتي حالياً. بنصحك تستشير مختص عشان تكون متطمن أكثر"

        context_str = ""
        context = related.sample(n=min(3, len(related)))[['Question', 'Answer']].to_dict('records')
        for item in context: context_str += f"- {item['Question']}\n"
        
        prompt = f"""تصرف كـ "نفس بوت". الحالة: {category}. السياق: {context_str}. سجل: {chat_history}. المستخدم: "{patient_input}". جاوب بلهجة عامية دافئة، متعاطفة، ونصيحة عملية قصيرة. طوارئ: 911."""
        response = models['model'].generate_content(prompt)
        return category, response.text
    except: return "Unknown", "عذراً، حدث خطأ في الاتصال."

def retrain_model(original_data_path, new_data_df):
    try:
        if os.path.exists(original_data_path):
            if original_data_path.endswith('.zip'):
                with zipfile.ZipFile(original_data_path, 'r') as z:
                    with z.open([n for n in z.namelist() if n.endswith('.pkl')][0]) as f: df_old = pd.read_pickle(f)
            else: df_old = pd.read_pickle(original_data_path)
        else: return False, "ملف البيانات الأصلي غير موجود"

        new_data_df = new_data_df[['Question', 'Answer', 'Hierarchical Diagnosis']]
        df_combined = pd.concat([df_old, new_data_df]).drop_duplicates(subset=['Question']).reset_index(drop=True)
        stemmer = ISRIStemmer()
        df_combined['processed'] = df_combined['Question'].apply(lambda x: " ".join([stemmer.stem(w) for w in str(x).split()]))
        
        cv, le = CountVectorizer(), LabelEncoder()
        X = cv.fit_transform(df_combined['processed']).toarray()
        y = le.fit_transform(df_combined['Hierarchical Diagnosis'])
        
        clf = SVC(kernel='linear'); clf.fit(X, y)
        with open('svm_model.pkl', 'wb') as f: pickle.dump(clf, f)
        with open('vectorizer.pkl', 'wb') as f: pickle.dump(cv, f)
        with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)
        df_combined.to_pickle('dataset_original.pkl')
        
        with zipfile.ZipFile('svm_model.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write('svm_model.pkl')
        with zipfile.ZipFile('dataset_original.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write('dataset_original.pkl')
        return True, f"تم التدريب بنجاح! عدد البيانات: {len(df_combined)}"
    except Exception as e: return False, str(e)

# ==========================================
# 4. قسم الواجهة والشات (UI & Chat)
# ==========================================
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap'); html, body, [class*='css'] { font-family: 'Cairo', sans-serif; } .stApp { background-color: #F0F2F5; } .user-msg { background-color: #FFFFFF; color: #333333; border: 1px solid #E0E0E0; padding: 10px; border-radius: 15px; margin: 5px; float: right; direction: rtl; } .bot-msg { background-color: #E3F2FD; color: #0D47A1; padding: 10px; border-radius: 15px; margin: 5px; float: left; direction: rtl; } .stButton>button { background-color: #1976D2 !important; color: white !important; border-radius: 8px; }</style>", unsafe_allow_html=True)

def main():
    if 'db' not in st.session_state: st.session_state.db = init_database()
    if 'models' not in st.session_state: st.session_state.models = load_nafsbot_models()
    if st.session_state.models is None: st.error("⚠️ ملفات النظام مفقودة!"); st.stop()
    if 'logged_in' not in st.session_state: st.session_state.update({'logged_in': False, 'user_id': None, 'username': None, 'current_session_id': None})

    conn = st.session_state.db

    if 'auto_train_check' not in st.session_state:
        try:
            new_df = get_new_training_data(conn)
            if len(new_df) > 5: 
                success, msg = retrain_model('dataset_original.zip', new_df)
                if success: st.session_state.models = load_nafsbot_models()
        except: pass
        st.session_state.auto_train_check = True

    if not st.session_state['logged_in']:
        st.title("🧠 نفس بوت الإلكتروني")
        tab1, tab2 = st.tabs(["🔐 تسجيل دخول", "👤 مستخدم جديد"])
        with tab1:
            u, p = st.text_input("اسم المستخدم", key="l_u"), st.text_input("كلمة المرور", type='password', key="l_p")
            if st.button("🚀 دخول"):
                res = login_user(conn, u, p)
                if res: st.session_state.update({'logged_in':True, 'user_id':res[0], 'username':res[1]}); st.rerun()
                else: st.error("خطأ في البيانات")
        with tab2:
            nu, np, np2 = st.text_input("مستخدم جديد", key="n_u"), st.text_input("كلمة مرور", type='password', key="n_p"), st.text_input("تأكيد", type='password', key="n_p2")
            if st.button("✨ إنشاء حساب"):
                if np!=np2: st.error("كلمتا المرور غير متطابقتين!")
                elif len(np)<6: st.error("كلمة المرور قصيرة.")
                else:
                    s, m = create_user(conn, nu, np)
                    if s: st.success(m)
                    else: st.warning(m)
    else:
        with st.sidebar:
            st.title(f"مرحباً, {st.session_state['username']} 👋")
            if st.button("➕ محادثة جديدة", type="primary"): st.session_state.current_session_id = None; st.rerun()
            st.markdown("---"); st.caption("📂 الأرشيف")
            for sid, title, date in get_user_sessions(conn, st.session_state.user_id):
                btn_type = "primary" if sid == st.session_state.current_session_id else "secondary"
                if st.button(f"{date[:10]} | {title}", key=f"s_{sid}", type=btn_type): st.session_state.current_session_id = sid; st.rerun()
            st.markdown("---")
            if st.session_state.current_session_id:
                with st.expander("⚙️ خيارات الجلسة"):
                    new_t = st.text_input("تعديل الاسم")
                    if st.button("حفظ الاسم"): rename_session(conn, st.session_state.current_session_id, new_t); st.rerun()
                    if st.button("🗑️ حذف الجلسة"): delete_session(conn, st.session_state.current_session_id); st.session_state.current_session_id = None; st.rerun()
            if st.button("تسجيل خروج"): st.session_state.clear(); st.rerun()

        chat_context = ""
        if st.session_state.current_session_id:
            for q, a in get_session_messages(conn, st.session_state.current_session_id):
                st.markdown(f'<div class="user-msg">👤 {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-msg">🧠 {a}</div>', unsafe_allow_html=True)
                chat_context += f"User: {q}\nBot: {a}\n"
        else: st.info("💡 ابدأ محادثة جديدة...")

        if user_input := st.chat_input("اكتب هنا..."):
            st.markdown(f'<div class="user-msg">👤 {user_input}</div>', unsafe_allow_html=True)
            if st.session_state.current_session_id is None: st.session_state.current_session_id = create_new_session(conn, st.session_state.user_id, user_input)
            cat, ans = get_nafsbot_response(st.session_state.models, user_input, chat_context)
            if ans:
                st.markdown(f'<div class="bot-msg">🧠 {ans}</div>', unsafe_allow_html=True)
                save_message(conn, st.session_state.current_session_id, st.session_state.user_id, user_input, ans, cat)
                if chat_context=="": st.rerun()
            else: st.error("خطأ تقني")

if __name__ == "__main__": main()
