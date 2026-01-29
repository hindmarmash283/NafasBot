import pickle
import pandas as pd
import os
import google.generativeai as genai
import pyarabic.araby as araby
from nltk.stem.isri import ISRIStemmer
import zipfile
import streamlit as st
import re  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

@st.cache_resource
def load_nafsbot_models():
    # 🛑 مفتاح API
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCK1kMchDgsxFPDHU3t2hXhn-h6sDOnHho"
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-3-flash')
    
    stemmer = ISRIStemmer()

    # --- 🔥 دالة المعالجة المتقدمة (Normalization) ---
    def normalize_arabic_word(word):
        word = araby.strip_tatweel(word)      # إزالة التطويل (ـ)
        word = araby.strip_tashkeel(word)     # إزالة التشكيل
        word = re.sub(r'[إأآا]', 'ا', word)   # توحيد الألف
        word = re.sub(r'ى', 'ي', word)        # توحيد الياء
        word = re.sub(r'ؤ', 'ء', word)        # توحيد الهمزات
        word = re.sub(r'ئ', 'ء', word)
        word = re.sub(r'ة', 'ه', word)        # التاء المربوطة -> هاء
        word = re.sub(r'(.)\1{2,}', r'\1', word) # إزالة التكرار (مثل: اااهلا -> اهلا)
        return word
    
    # دالة التقطيع (تستخدم المعالجة أولاً)
    def stem_arabic_word(text):
        try:
            text = normalize_arabic_word(text) # أولاً: معالجة النص
            words = text.split()
            # ثانياً: استخراج الجذر
            return " ".join([stemmer.stem(word) for word in words])
        except: return text
    
    try:
        svm_model, df_data = None, None
        
        # تحميل SVM (حسب اسم الملف عندك: svm_model.zip)
        if os.path.exists('svm_model.zip'):
            with zipfile.ZipFile('svm_model.zip', 'r') as z:
                pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f: svm_model = pickle.load(f)
        elif os.path.exists('svm_model.pkl'):
            with open('svm_model.pkl', 'rb') as f: svm_model = pickle.load(f)

        # تحميل Dataset (حسب اسم الملف عندك: dataset_original.zip)
        if os.path.exists('dataset_original.zip'):
            with zipfile.ZipFile('dataset_original.zip', 'r') as z:
                pkl_files = [n for n in z.namelist() if n.endswith('.pkl')]
                if pkl_files:
                    with z.open(pkl_files[0]) as f: df_data = pd.read_pickle(f)
        elif os.path.exists('dataset_original.pkl'):
            df_data = pd.read_pickle('dataset_original.pkl')

        # تحميل الملفات الصغيرة
        if os.path.exists('vectorizer.pkl'):
            with open('vectorizer.pkl', 'rb') as f: vec = pickle.load(f)
        if os.path.exists('label_encoder.pkl'):
            with open('label_encoder.pkl', 'rb') as f: enc = pickle.load(f)
        
        if svm_model is None or df_data is None:
            return None

        return {'model': model, 'svm': svm_model, 'vectorizer': vec, 
                'encoder': enc, 'data': df_data, 'stem': stem_arabic_word}
    except Exception as e:
        return None

def get_nafsbot_response(models, patient_input,chat_history):
    try:
        processed = models['stem'](patient_input)
        vec = models['vectorizer'].transform([processed]).toarray()
        pred_idx = models['svm'].predict(vec)[0]
        category = models['encoder'].inverse_transform([pred_idx])[0]
        
        related = models['data'][models['data']['Hierarchical Diagnosis'] == category]
        if len(related) == 0:
            # إذا لم نجد أي بيانات طبية، نرفض الإجابة فوراً ولا نسأل Gemini
            return category, "ما عندي معلومة طبية دقيقة عن حالتك في قاعدة بياناتي حالياً. بنصحك تستشير مختص عشان تكون متطمن أكثر"
   
        context_str = ""
        if len(related) > 0:
            context = related.sample(n=min(3, len(related)))[['Question', 'Answer']].to_dict('records')
            for item in context: context_str += f"- {item['Question']}\n"
        
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
    5.  اذا كان هناك اي نوع من انواع نية الموت او ايذاء النفس او الانتحار اعطي اجابات تدعم للغاية ووفر رقم الطوارئ للدعم النفسي 0795785095 او الطوائ العامة911 في الاردن
    أنت "نفس بوت"، صديق ذكي ومساعد للدعم النفسي فقط.
    سجل المحادثة السابقة:
    {chat_history}
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
        return "Unknown", "عذراً، حدث خطأ في الاتصال."
    
def retrain_model(original_data_path, new_data_df):
    """دمج البيانات وإعادة تدريب الموديل"""
    try:
        # 1. تحميل البيانات الأصلية
        if os.path.exists(original_data_path):
            if original_data_path.endswith('.zip'):
                with zipfile.ZipFile(original_data_path, 'r') as z:
                    with z.open([n for n in z.namelist() if n.endswith('.pkl')][0]) as f:
                        df_old = pd.read_pickle(f)
            else:
                df_old = pd.read_pickle(original_data_path)
        else:
            return False, "ملف البيانات الأصلي غير موجود"

        # 2. الدمج (Merge)
        # التأكد من تطابق الأعمدة
        new_data_df = new_data_df[['Question', 'Answer', 'Hierarchical Diagnosis']]
        df_combined = pd.concat([df_old, new_data_df]).drop_duplicates(subset=['Question']).reset_index(drop=True)

        # 3. إعادة التدريب (Retraining)
        # المعالجة
        stemmer = ISRIStemmer()
        df_combined['processed'] = df_combined['Question'].apply(lambda x: advanced_arabic_processing(str(x)))
        
        # Vectorizer
        cv = CountVectorizer()
        X = cv.fit_transform(df_combined['processed']).toarray()
        
        # Encoder
        le = LabelEncoder()
        y = le.fit_transform(df_combined['Hierarchical Diagnosis'])
        
        # SVM Training
        clf = SVC(kernel='linear')
        clf.fit(X, y)

        # 4. حفظ الملفات الجديدة
        # حفظ الموديلات الصغيرة
        with open('svm_model.pkl', 'wb') as f: pickle.dump(clf, f)
        with open('vectorizer.pkl', 'wb') as f: pickle.dump(cv, f)
        with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)
        # حفظ الداتا سيت الجديدة
        df_combined.to_pickle('dataset_original.pkl')

        # ضغط الملفات الكبيرة (لـ GitHub)
        with zipfile.ZipFile('svm_model.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write('svm_model.pkl')
        with zipfile.ZipFile('dataset_original.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            z.write('dataset_original.pkl')
            
        return True, f"تم التدريب بنجاح! عدد البيانات الجديد: {len(df_combined)}"
    except Exception as e:
        return False, f"فشل التدريب: {str(e)}"
