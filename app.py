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

# الأيقونة: روبوت (عقل إلكتروني) والعنوان
st.set_page_config(page_title="NafasBot AI", page_icon="🤖", layout="wide")

# CSS لتطبيق الألوان المطلوبة (أزرق فاتح، أبيض، سكني)
st.markdown("""
<style>
    /* استيراد خط عربي جميل */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    /* خلفية التطبيق: سكني فاتح جداً */
    .stApp {
        background-color: #F0F2F5;
    }
    
    /* العناوين باللون الأزرق التقني */
    h1, h2, h3 {
       color: #1565C0 !important;
    }

    /* فقاعة رسالة المستخدم (يمين - أبيض مع إطار) */
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
    
    /* فقاعة رسالة البوت (يسار - أزرق فاتح) */
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
    
    /* تحسين الأزرار */
    .stButton>button {
        background-color: #1976D2 !important;
        color: white !important;
        border-radius: 8px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# إعداد قاعدة البيانات (مع التعديلات الجديدة)
# ============================================================

def init_database():
    """إنشاء قاعدة البيانات + التنظيف التلقائي"""
    conn = sqlite3.connect('nafasbot.db', check_same_thread=False)
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
    
    # 🔥 ميزة الحذف التلقائي: حذف أي رسالة أقدم من 15 يوم عند التشغيل
    fifteen_days_ago = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (fifteen_days_ago,))
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
    # الفلترة: لا نحفظ إذا كان التصنيف غير معروف (لضمان جودة البيانات)
    if category and category != "Unknown":
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(days=15)
        cursor.execute('''
            INSERT INTO conversations (user_id, question, answer, category, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, question, answer, category, expires_at))
        conn.commit()

def get_user_history(conn, user_id):
    """جلب المحادثات السابقة لعرضها"""
    cursor = conn.cursor()
    cursor.execute('SELECT question, answer FROM conversations WHERE user_id=? ORDER BY timestamp ASC', (user_id,))
    return cursor.fetchall()

# ============================================================
# تحميل NafsBot (نفس الكود القديم)
# ============================================================

@st.cache_resource
def load_nafsbot_models():
    """تحميل النماذج بذكاء"""
    
    # 🛑🛑🛑 تنبيه: تأكدي من وضع المفتاح هنا 🛑🛑🛑
    my_api_key = "AIzaSyBawgdx3fLKoY6MuLYugJiSPazVK54GG_s"
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
        # 1. تحميل SVM
        svm_model = None
        if os.path.exists('svm_model.zip'):
            with zipfile.ZipFile('svm_model.zip', 'r') as z:
                pkl_files = [f for f in z.namelist() if f.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f:
                        svm_model = pickle.load(f)
        
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
# المحرك الرئيسي (نفس البرومبت الخاص بك)
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
        
        # التوليد (نفس البرومبت الذي طلبت الحفاظ عليه)
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
# الواجهة الرئيسية (التعديل الجذري هنا)
# ============================================================

def main():
    # 1. تهيئة النظام
    if 'db' not in st.session_state: st.session_state.db = init_database()
    if 'models' not in st.session_state: st.session_state.models = load_nafsbot_models()
    
    # متغيرات الجلسة (تسجيل الدخول)
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_id'] = None
        st.session_state['username'] = None

    conn = st.session_state.db

    # 2. السيناريو الأول: المستخدم غير مسجل دخول
    if not st.session_state['logged_in']:
        st.title("🧠 نفس بوت الإلكتروني")
        st.markdown("### مساحتك الآمنة للفضفضة والدعم النفسي")
        
        tab1, tab2 = st.tabs(["🔐 تسجيل دخول", "👤 مستخدم جديد"])
        
        with tab1:
            username = st.text_input("اسم المستخدم", key="login_user")
            password = st.text_input("كلمة المرور", type='password', key="login_pass")
            st.write("")
            if st.button("🚀 دخول"):
                result = login_user(conn, username, password)
                if result[0]: # نجاح
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = result[1]
                    st.session_state['username'] = result[2]
                    st.rerun()
                else:
                    st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
                    
        with tab2:
            new_user = st.text_input("اختر اسم مستخدم", key="new_user")
            new_pass = st.text_input("اختر كلمة مرور", type='password', key="new_pass")
            st.write("")
            if st.button("✨ إنشاء حساب"):
                success, msg = create_user(conn, new_user, new_pass)
                if success:
                    st.success(msg + " .. يمكنك الآن تسجيل الدخول")
                else:
                    st.warning(msg)

    # 3. السيناريو الثاني: المستخدم مسجل دخول (الشات)
    else:
        # القائمة الجانبية
        with st.sidebar:
            st.title(f"أهلاً, {st.session_state['username']} 🧠")
            st.markdown("---")
            if st.button("تسجيل خروج"):
                st.session_state['logged_in'] = False
                st.rerun()
            st.markdown("---")
            st.info("🔒 المحادثات آمنة ومحفوظة لمدة 15 يوماً فقط.")

        st.title("💬 جلسة نفسية ذكية")
        
        # عرض المحادثات السابقة من قاعدة البيانات (تصميم الواتساب)
        history = get_user_history(conn, st.session_state['user_id'])
        for q, a in history:
            st.markdown(f'<div class="user-msg">👤 {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-msg">🧠 {a}</div>', unsafe_allow_html=True)
            
        # إدخال رسالة جديدة
        if user_input := st.chat_input("تحدث معي... أنا هنا لأسمعك"):
            # عرض الرسالة فوراً
            st.markdown(f'<div class="user-msg">👤 {user_input}</div>', unsafe_allow_html=True)
            
            # معالجة الرد
            cat, ans = get_nafsbot_response(st.session_state.models, user_input)
            
            # عرض الرد
            if ans:
                st.markdown(f'<div class="bot-msg">🧠 {ans}</div>', unsafe_allow_html=True)
                # حفظ في قاعدة البيانات
                save_conversation(conn, st.session_state['user_id'], user_input, ans, cat)
            else:
                st.error("حدث خطأ في الاتصال")

if __name__ == "__main__":
    main()
