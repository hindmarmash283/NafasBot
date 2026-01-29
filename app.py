import streamlit as st
import db_manager as db
import ai_engine as ai

# إعدادات الصفحة
st.set_page_config(page_title="NafasBot AI", page_icon="🤖", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #F0F2F5; }
    h1, h2, h3 { color: #1565C0 !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-left: 1px solid #E0E0E0; }
    .user-msg { background-color: #FFFFFF; color: #333333; border: 1px solid #E0E0E0; padding: 10px; border-radius: 15px; margin: 5px; float: right; direction: rtl; }
    .bot-msg { background-color: #E3F2FD; color: #0D47A1; padding: 10px; border-radius: 15px; margin: 5px; float: left; direction: rtl; }
    .stButton>button { background-color: #1976D2 !important; color: white !important; border-radius: 8px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def main():
    if 'db' not in st.session_state: st.session_state.db = db.init_database()
    
    # تحميل المودلز
    if 'models' not in st.session_state: 
        st.session_state.models = ai.load_nafsbot_models()

    # إذا فشل التحميل، أوقف البرنامج واعرض رسالة خطأ بدلاً من الانهيار لاحقاً
    if st.session_state.models is None:
        st.error("⚠️ عذراً، هناك مشكلة في ملفات النظام (nafas_model.zip). يرجى التأكد من رفعها على GitHub.")
        st.stop()
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_id'] = None
        st.session_state['username'] = None
        st.session_state['current_session_id'] = None

    conn = st.session_state.db

    if 'auto_train_check' not in st.session_state:
        try:
            # نتحقق هل يوجد بيانات جديدة؟
            new_df = db.get_new_training_data(conn)
            # الشرط: إذا كان هناك أكثر من 5 محادثات جديدة، ابدأ التدريب
            if len(new_df) > 5: 
                success, msg = ai.retrain_model('nafas_data.zip', new_df)
                if success:
                    print("✅ Auto-Training Successful!") # يظهر في الكونسول للمطور فقط
                    st.session_state.models = ai.load_nafsbot_models() # تحديث الموديل في الذاكرة
        except Exception as e:
            print(f"⚠️ Auto-Training Skipped: {e}")
        
        st.session_state.auto_train_check = True

    # --- تسجيل الدخول ---
    if not st.session_state['logged_in']:
        st.title("🧠 نفس بوت الإلكتروني")
        tab1, tab2 = st.tabs(["🔐 تسجيل دخول", "👤 مستخدم جديد"])
        
        with tab1:
            u = st.text_input("اسم المستخدم", key="l_u")
            p = st.text_input("كلمة المرور", type='password', key="l_p")
            if st.button("🚀 دخول"):
                res = db.login_user(conn, u, p)
                if res:
                    st.session_state['logged_in'] = True
                    st.session_state['user_id'] = res[0]
                    st.session_state['username'] = res[1]
                    st.rerun()
                else: st.error(" خطأ في اسم المستخدم أو كلمة المرور")
        
        with tab2:
            nu = st.text_input("اختر اسم مستخدم", key="n_u")
            np = st.text_input("اختر كلمة مرور", type='password', key="n_p")
            np2 = st.text_input("تأكيد كلمة المرور", type='password', key="n_p2") # حقل جديد
            
            if st.button("✨ إنشاء حساب"):
                if np != np2:
                    st.error("⚠️ كلمتا المرور غير متطابقتين!")
                elif len(np) < 6:
                    st.error("⚠️ كلمة المرور يجب أن تكون 6 أحرف على الأقل.")
                else:
                    # إذا نجحت الشروط، ننشئ الحساب
                    suc, msg = db.create_user(conn, nu, np)
                    if suc: st.success(msg)
                    else: st.warning(msg)


    # --- النظام الداخلي ---
    else:
        with st.sidebar:
            st.title(f"مرحباً, {st.session_state['username']} 👋")
            
            # 1. زر محادثة جديدة
            if st.button("➕ محادثة جديدة", type="primary"):
                st.session_state.current_session_id = None
                st.rerun()
            
            st.markdown("---")
            st.caption("📂 الأرشيف")
            
            # 2. قائمة الجلسات
            sessions = db.get_user_sessions(conn, st.session_state.user_id)
            for sess in sessions:
                sid, title, date = sess
                # تلوين الجلسة النشطة
                btn_type = "primary" if sid == st.session_state.current_session_id else "secondary"
                if st.button(f"{date[:10]} | {title}", key=f"s_{sid}", type=btn_type):
                    st.session_state.current_session_id = sid
                    st.rerun()

            st.markdown("---")
            
            # 3. إعدادات الجلسة الحالية
            if st.session_state.current_session_id:
                with st.expander("⚙️ خيارات الجلسة"):
                    new_t = st.text_input("تعديل الاسم")
                    if st.button("حفظ الاسم"):
                        db.rename_session(conn, st.session_state.current_session_id, new_t)
                        st.rerun()
                    if st.button("🗑️ حذف الجلسة"):
                        db.delete_session(conn, st.session_state.current_session_id)
                        st.session_state.current_session_id = None
                        st.rerun()

        # 5. تسجيل الخروج
            st.markdown("---")
            if st.button("تسجيل خروج"):
                st.session_state.clear()
                st.rerun()

        # --- منطقة الشات ---
        # إذا لم يكن هناك جلسة، ننشئ واحدة عند أول رسالة
        chat_context = ""
        if st.session_state.current_session_id:
            msgs = db.get_session_messages(conn, st.session_state.current_session_id)
            chat_context = ""
            for q, a in msgs:
                st.markdown(f'<div class="user-msg">👤 {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-msg">🧠 {a}</div>', unsafe_allow_html=True)
                chat_context += f"User: {q}\nBot: {a}\n"
        else:
            st.info("💡 ابدأ محادثة جديدة...")
            chat_context = ""

        if user_input := st.chat_input("اكتب هنا..."):
            st.markdown(f'<div class="user-msg">👤 {user_input}</div>', unsafe_allow_html=True)
            
            # إنشاء جلسة إذا كانت أول رسالة
            is_new_session = False
            if st.session_state.current_session_id is None:
                st.session_state.current_session_id = db.create_new_session(conn, st.session_state.user_id, user_input)
                is_new_session = True
            
            # الرد   
            cat, ans = ai.get_nafsbot_response(st.session_state.models, user_input, chat_context)         
            if ans:
                st.markdown(f'<div class="bot-msg">🧠 {ans}</div>', unsafe_allow_html=True)
                db.save_message(conn, st.session_state.current_session_id, st.session_state.user_id, user_input, ans, cat)
                if is_new_session:
                    st.rerun()
            else:
                st.error("خطأ تقني في الاتصال")

if __name__ == "__main__":
    main()
