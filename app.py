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
import zipfile  # مكتبة جديدة لفك الضغط

# ============================================================
# إعداد قاعدة البيانات
# ============================================================

def init_database():
    """إنشاء قاعدة البيانات"""
    conn = sqlite3.connect('nafsbot.db', check_same_thread=False)
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
    CREATE TABLE IF NOT EXISTS conversations (
        conv_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        category TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    conn.commit()
    return conn

# ============================================================
# دوال المستخدمين
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(conn, username, password):
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, hash_password(password))
        )
        conn.commit()
        return True, "تم إنشاء الحساب بنجاح!"
    except sqlite3.IntegrityError:
        return False, "اسم المستخدم موجود مسبقاً"

def login_user(conn, username, password):
    cursor = conn.cursor()
    cursor.execute(
        'SELECT user_id, username FROM users WHERE username = ? AND password_hash = ?',
        (username, hash_password(password))
    )
    result = cursor.fetchone()
    if result:
        return True, result[0], result[1]
    return False, None, None

def save_conversation(conn, user_id, question, answer, category):
    cursor = conn.cursor()
    expires_at = datetime.now() + timedelta(days=15)
    cursor.execute('''
        INSERT INTO conversations (user_id, question, answer, category, expires_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, question, answer, category, expires_at))
    conn.commit()

# ============================================================
# تحميل NafsBot (مع دعم الملفات المضغوطة)
# ============================================================

@st.cache_resource
def load_nafsbot_models():
    """تحميل النماذج بذكاء"""
    
    # إعداد Gemini
    # 🛑 تأكدي أن مفتاحك موجود هنا
    my_api_key = "AIzaSyBUbM_cKLyxHJfb_Ay8EGUc6FZ9PZuHS4I"
    os.environ["GOOGLE_API_KEY"] = my_api_key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    stemmer = ISRIStemmer()
    
    def stem_arabic_word(text):
        try:
            text = araby.strip_tashkeel(text)
            words = text.split()
            return " ".join([stemmer.stem(word) for word in words])
        except:
            return text
    
    try:
        # 1. تحميل SVM (البحث عن ملف pkl داخل الـ zip)
        svm_model = None
        if os.path.exists('svm_model.zip'):
            with zipfile.ZipFile('svm_model.zip', 'r') as z:
                # نبحث عن الملف الذي ينتهي بـ .pkl
                pkl_files = [f for f in z.namelist() if f.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f:
                        svm_model = pickle.load(f)
        
        # محاولة احتياطية (لو الملف مش مضغوط)
        if svm_model is None and os.path.exists('svm_model.pkl'):
            with open('svm_model.pkl', 'rb') as f:
                svm_model = pickle.load(f)

        if svm_model is None:
            raise Exception("لم يتم العثور على ملف الموديل svm_model")

        # 2. تحميل Dataset
        df_data = None
        if os.path.exists('dataset_original.zip'):
            with zipfile.ZipFile('dataset_original.zip', 'r') as z:
                pkl_files = [f for f in z.namelist() if f.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f:
                        df_data = pd.read_pickle(f)
        
        if df_data is None and os.path.exists('dataset_original.pkl'):
            df_data = pd.read_pickle('dataset_original.pkl')

        if df_data is None:
             raise Exception("لم يتم العثور على ملف البيانات dataset")

        # 3. الملفات الصغيرة
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return {
            'model': model,
            'svm': svm_model,
            'vectorizer': vectorizer,
            'encoder': label_encoder,
            'data': df_data,
            'stem': stem_arabic_word
        }
    except Exception as e:
        st.error(f"⚠️ خطأ في تشغيل النظام: {e}")
        return None

# ============================================================
# المحرك الرئيسي
# ============================================================

def get_nafsbot_response(models, patient_input):
    try:
        # التصنيف
        processed = models['stem'](patient_input)
        vec = models['vectorizer'].transform([processed]).toarray()
        pred_idx = models['svm'].predict(vec)[0]
        category = models['encoder'].inverse_transform([pred_idx])[0]
        
        # السياق
        related = models['data'][models['data']['Hierarchical Diagnosis'] == category]
        context_str = ""
        if len(related) > 0:
            context = related.sample(n=min(3, len(related)))[['Question', 'Answer']].to_dict('records')
            for item in context:
                context_str += f"- {item['Question']}\n"
        
        # التوليد
        prompt = f"""
    تصرف كـ "نفس بوت"، صديق مقرب وداعم نفسي حكيم.
    المستخدم بيمر بحالة تم تصنيفها كـ: {category}

    إليك بعض الحالات السابقة كمرجع (خذ منها الفائدة بس لا تنسخها):
    {context_str}

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
# الواجهة
# ============================================================

def main():
    st.set_page_config(page_title="نفس بوت", page_icon="🧠", layout="wide")
    st.markdown("<style>.main {direction: rtl;}</style>", unsafe_allow_html=True)
    
    if 'db' not in st.session_state: st.session_state.db = init_database()
    if 'models' not in st.session_state: st.session_state.models = load_nafsbot_models()
    if 'messages' not in st.session_state: st.session_state.messages = []
    
    st.title("🧠 نفس بوت")
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if user_input := st.chat_input("تحدث معي..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        cat, ans = get_nafsbot_response(st.session_state.models, user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.chat_message("assistant").write(ans)

if __name__ == "__main__":
    main()
