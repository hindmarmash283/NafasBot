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

# ============================================================
# 1. إعدادات الصفحة والتصميم (Blue & Grey Tech Theme) 🎨
# ============================================================

st.set_page_config(page_title="NafasBot AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    .stApp {
        background-color: #F0F2F5;
    }
    
    h1, h2, h3 {
       color: #1565C0 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-left: 1px solid #E0E0E0;
    }

    .user-msg {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #E0E0E0;
        padding: 10px 15px;
        border-radius: 15px 15px 2px 15px;
        margin: 5px;
        text-align: right;
        direction: rtl;
        float: right;
        width: fit-content;
        max-width: 75%;
        clear: both;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    
    .bot-msg {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 2px;
        margin: 5px;
        text-align: right;
        direction: rtl;
        float: left;
        width: fit-content;
        max-width: 75%;
        clear: both;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
        border: 1px solid #BBDEFB;
    }
    
    .stButton>button {
        background-color: #1976D2 !important;
        color: white !important;
        border-radius: 8px;
        width: 100%;
    }
    
    /* تنسيق خاص لزر الحذف ليكون مميزاً */
    div[data-testid="stExpander"] button {
         border: 1px solid #ff4b4b;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. إعداد قاعدة البيانات 🗄️
# ============================================================

def init_database():
    """إنشاء الجداول بنظام الجلسات"""
    conn = sqlite3.connect('nafasbot_sessions.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
    ''')
    
    # التنظيف التلقائي
    fifteen_days_ago = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("DELETE FROM sessions WHERE created_at < ?", (fifteen_days_ago,))
    
    conn.commit()
    return conn

# --- دوال إدارة المستخدمين ---
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
    result = cursor.fetchone()
    if result:
        return True, result[0], result[1]
    return False, None, None

# --- دوال إدارة الجلسات والرسائل (مع الحذف والتعديل) ---

def create_new_session(conn, user_id, first_msg):
    cursor = conn.cursor()
    title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
    cursor.execute('INSERT INTO sessions (user_id, title) VALUES (?, ?)', (user_id, title))
    conn.commit()
    return cursor.lastrowid

def get_user_sessions(conn, user_id):
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, title, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC', (user_id,))
    return cursor.fetchall()

def save_message(conn, session_id, user_id, q, a, cat):
    if cat and cat != "Unknown":
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (session_id, user_id, question, answer, category)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, user_id, q, a, cat))
        conn.commit()

def get_session_messages(conn, session_id):
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM messages WHERE session_id=? ORDER BY timestamp ASC', (session_id,))
    return cursor.fetchall()

# 🔥 دالة حذف الجلسة بالكامل
def delete_session(conn, session_id):
    cursor = conn.cursor()
    # نحذف الرسائل أولاً (لتنظيف النظام)
    cursor.execute('DELETE FROM messages WHERE session_id=?', (session_id,))
    # ثم نحذف الجلسة نفسها
    cursor.execute('DELETE FROM sessions WHERE session_id=?', (session_id,))
    conn.commit()

# 🔥 دالة تعديل اسم الجلسة
def rename_session(conn, session_id, new_title):
    cursor = conn.cursor()
    cursor.execute('UPDATE sessions SET title=? WHERE session_id=?', (new_title, session_id))
    conn.commit()

# ============================================================
# 3. تحميل NafsBot
# ============================================================

@st.cache_resource
def load_nafsbot_models():
    # 🛑🛑🛑 ضعي مفتاحك هنا 🛑🛑🛑
    my_api_key = "AIzaSyCK1kMchDgsxFPDHU3t2hXhn-h6sDOnHho" 
    os.environ["GOOGLE_API_KEY"] = my_api_key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    stemmer = ISRIStemmer()
    def stem_arabic_word(text):
        try:
            text = araby.strip_tashkeel(text)
            words = text.split()
            return " ".join([stemmer.stem(word) for word in words])
        except: return text
    
    try:
        nafas_model, df_data = None, None
        
        # Load SVM
        if os.path.exists('nafas_model.zip'):
            with zipfile.ZipFile('nafas_model.zip', 'r') as z:
                with z.open([n for n in z.namelist() if n.endswith('.pkl')][0]) as f: 
                    nafas_model = pickle.load(f)
        elif os.path.exists('nafas_model.pkl'):
            with open('nafas_model.pkl', 'rb') as f: nafas_model = pickle.load(f)

        # Load Dataset
        if os.path.exists('nafas_data.zip'):
            with zipfile.ZipFile('nafas_data.zip', 'r') as z:
                with z.open([n for n in z.namelist() if n.endswith('.pkl')][0]) as f: 
                    df_data = pd.read_pickle(f)
        elif os.path.exists('nafas_data.pkl'):
            df_data = pd.read_pickle('nafas_data.pkl')

        with open('vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: enc = pickle.load(f)
        
        return {'model': model, 'svm': nafas_model, 'vectorizer': vec, 
                'encoder': enc, 'data': df_data, 'stem': stem_arabic_word}
    except Exception as e:
        st.error(f"⚠️ خطأ: {e}")
        return None

# ============================================================
# المحرك الرئيسي
# ============================================================

def get_nafsbot_response(models, patient_input ,chat_history):
    try:
        processed = models['stem'](patient_input)
        vec = models['vectorizer'].transform([processed]).toarray()
        pred_idx = models['svm'].predict(vec)[0]
        category = models['encoder'].inverse_transform([pred_idx])[0]
        
        related = models['data'][models['data']['Hierarchical Diagnosis'] == category]
        context_str = ""
        if len(related) > 0:
            context = related.sample(n=min(3, len(related)))[['Question', 'Answer']].to_dict('records')
            for item in context:
                context_str += f"- {item['Question']}\n"
        
        prompt = f"""
    تصرف كـ "نفس بوت"، صديق مقرب وداعم نفسي حكيم.
    المستخدم بيمر بحالة تم تصنيفها كـ: {category}

    إليك بعض الحالات السابقة كمرجع (خذ منها الفائدة بس لا تنسخها):
    {context_str}
    
    سجل المحادثة الحالية بينك وبين المستخدم (عشان تربط الكلام ببعضه):
    {chat_history}
    المستخدم بيحكيلك: "{patient_input}"

    المطلوب منك:
    1. رد عليه بلهجة عامية بيضاء (قريبة للكل) وبأسلوب "صديق لصديقه".
    2. كون متعاطف جداً، طمنه، وحسسه إنك جنبه وسامعه.
    3. أعطيه نصيحة بسيطة وعملية بناءً على السياق الطبي بس بكلمات بسيطة مش معقدة.
    4. خلي الرد قصير ومباشر (من 3 لـ 4 جمل) ولازم تربط الرسائل اللي بحكيلك اياها المريق ببعض ما تجاوب على كل رسالة بشكل منفصل حاول تربط المواضيع
    5.  اذا كان هناك اي نوع من انواع نية الموت او ايذاء النفس او الانتحار اعطي اجابات تدعم للغاية ووفر رقم الطوارئ للدعم النفسي 0795785095 او الطوائ العامة911 في الاردن
    أنت "نفس بوت"، صديق ذكي ومساعد للدعم النفسي فقط.

    تعليمات صارمة ومهمة جداً:
    1. اقرأ رسالة المستخدم جيداً: "{patient_input}"
    2. حدد الموضوع:
       - إذا كان الكلام عن مشاعر، ضيق، خوف، اكتئاب، فضفضة، أو تحية (مرحبا، كيفك): كمل وجاوب كصديق داعم.
       - إذا كان الكلام عن (طبخ، رياضة، سياسة، حل واجبات، معلومات عامة، بيع وشراء): **توقف فوراً**.

    3. في حالة السؤال الخارجي (غير نفسي):
       - اعتذر بلطف شديد وبالعامية.
       - قل له جملة بمعني: "سامحني يا غالي، أنا هون بس عشان أسمعك وأدعمك نفسياً، ما عندي خبرة بهيك مواضيع".
       - لا تجب على السؤال أبداً.

    4. في حالة الكلام النفسي أو الفضفضة:
       - تصنيف الحالة: {category}
       - السياق الطبي للمساعدة: {context_str}
       - رد عليه بلهجة عامية بيضاء، بأسلوب صديق مقرب وحكيم، وطمنه.
    """
        response = models['model'].generate_content(prompt)
        return category, response.text
    except Exception as e:
        return None, f"خطأ: {str(e)}"

# ============================================================
# الواجهة الرئيسية (مع ميزات الحذف والتعديل)
# ============================================================

def main():
    if 'db' not in st.session_state: st.session_state.db = init_database()
    if 'models' not in st.session_state: st.session_state.models = load_nafsbot_models()
    
    if 'logged_in' not in st.session_state:
        st.session_state.update({'logged_in': False, 'user_id': None, 'username': None, 'current_session_id': None})

    conn = st.session_state.db

    # --- تسجيل الدخول ---
    if not st.session_state['logged_in']:
        st.title("🧠 نفس بوت الإلكتروني")
        st.markdown("### مساحتك الآمنة للفضفضة والدعم النفسي")
        tab1, tab2 = st.tabs(["🔐 تسجيل دخول", "👤 مستخدم جديد"])
        with tab1:
            u = st.text_input("اسم المستخدم", key="l_u")
            p = st.text_input("كلمة المرور", type="password", key="l_p")
            if st.button("🚀 دخول"):
                res = login_user(conn, u, p)
                if res[0]:
                    st.session_state.logged_in = True
                    st.session_state.user_id = res[1]
                    st.session_state.username = res[2]
                    st.rerun()
                else: st.error("خطأ في البيانات")
        with tab2:
            nu = st.text_input("اختر اسم مستخدم", key="n_u")
            np = st.text_input("اختر كلمة مرور", type="password", key="n_p")
            if st.button("✨ إنشاء حساب"):
                suc, msg = create_user(conn, nu, np)
                if suc: st.success(msg)
                else: st.warning(msg)

    # --- النظام الداخلي ---
    else:
        with st.sidebar:
            st.title(f"أهلاً, {st.session_state.username} 👋")
            
            if st.button("➕ محادثة جديدة", type="primary"):
                st.session_state.current_session_id = None
                st.rerun()
            
            st.markdown("---")
            
            # 🔥 إعدادات المحادثة الحالية (تظهر فقط عند فتح محادثة)
            if st.session_state.current_session_id:
                with st.expander("⚙️ إعدادات المحادثة الحالية", expanded=True):
                    # تعديل الاسم
                    new_title = st.text_input("تغيير الاسم:", placeholder="اكتب الاسم الجديد...")
                    if st.button("💾 حفظ الاسم"):
                        if new_title:
                            rename_session(conn, st.session_state.current_session_id, new_title)
                            st.success("تم التعديل!")
                            st.rerun()
                    
                    st.markdown("---")
                    # حذف الجلسة
                    if st.button("🗑️ حذف المحادثة نهائياً"):
                        delete_session(conn, st.session_state.current_session_id)
                        st.session_state.current_session_id = None # الخروج للصفحة الرئيسية
                        st.rerun()

            st.markdown("---")
            st.caption("🗂️ الأرشيف")
            sessions = get_user_sessions(conn, st.session_state.user_id)
            for sess in sessions:
                sess_id, title, date = sess
                display_date = date.split()[0]
                if st.button(f"{display_date} | {title}", key=f"btn_{sess_id}"):
                    st.session_state.current_session_id = sess_id
                    st.rerun()
            
            st.markdown("---")
            if st.button("تسجيل خروج"):
                st.session_state.clear()
                st.rerun()

        # منطقة الشات
        if st.session_state.current_session_id:
            msgs = get_session_messages(conn, st.session_state.current_session_id)
            for q, a in msgs:
                st.markdown(f'<div class="user-msg">👤 {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-msg">🧠 {a}</div>', unsafe_allow_html=True)
        else:
            st.info("💡 ابدأ محادثة جديدة..")

        if user_input := st.chat_input("اكتب ما تشعر به..."):
            st.markdown(f'<div class="user-msg">👤 {user_input}</div>', unsafe_allow_html=True)
            
            if st.session_state.current_session_id is None:
                new_sess_id = create_new_session(conn, st.session_state.user_id, user_input)
                st.session_state.current_session_id = new_sess_id
            
            # --- 👇 هنا التعديل الجديد: تحضير الذاكرة ---
            # 1. نجلب الرسائل السابقة لهذه الجلسة
            prev_msgs = get_session_messages(conn, st.session_state.current_session_id)
            history_str = ""
            # 2. ندمجها في نص واحد (نأخذ آخر 5 رسائل لتوفير التوكنز)
            for q, a in prev_msgs[-5:]:
                history_str += f"المستخدم: {q}\nنفس بوت: {a}\n"
            
            # 3. نرسل الـ history_str للدالة
            cat, ans = get_nafsbot_response(st.session_state.models, user_input, history_str)
            # --- 👆 نهاية التعديل ---
            
            if ans:
                st.markdown(f'<div class="bot-msg">🧠 {ans}</div>', unsafe_allow_html=True)
                save_message(conn, st.session_state.current_session_id, st.session_state.user_id, user_input, ans, cat)
            else:
                st.error("خطأ في الاتصال")
if __name__ == "__main__":
    main()
