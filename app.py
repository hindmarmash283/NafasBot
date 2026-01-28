import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import re
import pyarabic.araby as araby
from nltk.stem.isri import ISRIStemmer

# ============================================================
# ⚙️ إعدادات الصفحة (Streamlit Configuration)
# ============================================================
st.set_page_config(
    page_title="نفس بوت | NafasBot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# تنسيق اتجاه النص لليمين (RTL) لأن التطبيق عربي
st.markdown("""
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 🔑 الخطوة 1: مفتاح الربط (API Key)
# ============================================================
# تم الحفاظ على المفتاح كما طلبتِ
my_api_key = "AIzaSyCgc326bDm51rHLS6CSDCLfzoQ1Y6Yg0b4"

os.environ["GOOGLE_API_KEY"] = my_api_key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# استخدام الموديل المحدد
model = genai.GenerativeModel('gemini-2.5-flash')

# ============================================================
# 🧠 الخطوة 2: تحميل "دماغ" النظام (Caching لتسريع التطبيق)
# ============================================================
# نستخدم st.cache_resource عشان ما يحمل الملفات مع كل رسالة جديدة
@st.cache_resource
def load_nafsbot_brain():
    try:
        with open('svm_model.pkl', 'rb') as f: loaded_model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f: loaded_vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: loaded_encoder = pickle.load(f)
        df_data = pd.read_pickle('dataset_original.pkl')
        return loaded_model, loaded_vectorizer, loaded_encoder, df_data
    except FileNotFoundError:
        return None, None, None, None

loaded_model, loaded_vectorizer, loaded_encoder, df_data = load_nafsbot_brain()

# التحقق من تحميل الملفات
if loaded_model is None:
    st.error("❌ خطأ: لم يتم العثور على ملفات الذكاء (.pkl). تأكدي من وجود الملفات بجانب ملف app.py")
    st.stop()

# ============================================================
# 🛠️ دوال المعالجة (Pre-processing)
# ============================================================
stemmer = ISRIStemmer()

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
    text = normalize_arabic_word(text)
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

# ============================================================
# 🔗 الخطوة 3: دالة التكامل (The Integration Logic)
# ============================================================
def get_nafsbot_response(patient_input):
    """
    نفس الدالة الأصلية تماماً مع الحفاظ على البرومبت والتعليمات
    """
    # أ. التصنيف باستخدام الموديل المحلي (SVM)
    try:
        processed_text = stem_arabic_word(patient_input)
        vec = loaded_vectorizer.transform([processed_text]).toarray()
        prediction_idx = loaded_model.predict(vec)[0]
        category = loaded_encoder.inverse_transform([prediction_idx])[0]
    except:
        return "عذراً، حدث خطأ تقني في تحليل النص."

    # ب. جلب السياق من بياناتك (Retrieval)
    if df_data is not None:
        related_data = df_data[df_data['Hierarchical Diagnosis'] == category]
        if not related_data.empty:
            context = related_data.sample(n=min(3, len(related_data)))[['Question', 'Answer']].to_dict('records')
        else:
            context = []
    else:
        context = []

    # ج. تجهيز البرومبت
    context_str = ""
    for item in context:
        context_str += f"- حالة سابقة: {item['Question'][:100]}...\n- الإجراء الطبي: {item['Answer'][:200]}...\n\n"

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
    4. خلي الرد قصير ومباشر (من 3 لـ 4 جمل).
    5. أضف مثل شعبي عربي أو فلسطيني زي 'الصبر جميل' أو 'ما بعد الضيق إلا الفرج' لتشجيع الصبر والأمل.
    6. اذا كان هناك اي نوع من انواع نية الموت او ايذاء النفس او الانتحار اعطي اجابات تدعم للغاية ووفر رقم الطوارئ للدعم النفسي 0795785095 او الطوائ العامة911 في الاردن
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

    # د. الاتصال بـ Gemini
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"عذراً، حدثت مشكلة في الاتصال: {e}"

# ============================================================
# 💻 واجهة التطبيق (UI Application)
# ============================================================

st.title("🤖 نفس بوت | NafasBot")
st.markdown("### رفيقك الذكي للدعم النفسي 💙")
st.caption("فضفض وأنا بسمعك.. مساحتك الآمنة للحديث.")

# 1. تهيئة سجل المحادثة (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. عرض الرسائل القديمة في الشاشة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. استقبال مدخلات المستخدم (Chat Input)
if prompt := st.chat_input("بماذا تشعر اليوم؟"):
    # عرض رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # معالجة الرد من البوت
    with st.chat_message("assistant"):
        with st.spinner("نفس بوت يكتب..."):
            response_text = get_nafsbot_response(prompt)
            st.markdown(response_text)
    
    # حفظ رد البوت في السجل
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# إضافة زر لمسح المحادثة في القائمة الجانبية
with st.sidebar:
    st.info("💡 هذا بوت تجريبي لمساعدة مشروع التخرج.")
    if st.button("بدء محادثة جديدة"):
        st.session_state.messages = []
        st.rerun()
