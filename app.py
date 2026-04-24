import streamlit as st
import google.generativeai as genai
from retriever2 import get_relevant_course_context, get_page_image, calculate_hallucination_score
import urllib.request
import urllib.parse
import re
import os

# --- SECRETS & CONFIG ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash') # Uses the stable API version

# Premium Allowed Accounts (Username: Password)
PREMIUM_ACCOUNTS = {
    "anwesh": "admin123",
    "professor": "bitlearn2024",
    "vip_student": "pass"
}

st.set_page_config(layout="wide", page_title="BITlearn")
st.title("BITlearn")

@st.dialog("Textbook Scan")
def show_page_pop(source, page):
    st.write(f"**{source}** | Page {page}")
    img = get_page_image(source, page)
    if img:
        st.image(img, use_container_width=True)
    else:
        st.error("Failed to render page.")

def get_yt_videos(query):
    try:
        search_keyword = urllib.parse.quote(f"{query} computer science")
        url = f"https://www.youtube.com/results?search_query={search_keyword}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        
        unique_ids = []
        for vid in video_ids:
            if vid not in unique_ids:
                unique_ids.append(vid)
                
        if unique_ids:
            return [f"https://www.youtube.com/watch?v={vid}" for vid in unique_ids[:2]]
        return []
    except:
        return []

# --- SESSION STATE ---
if "data" not in st.session_state: st.session_state.data = None
if "ans" not in st.session_state: st.session_state.ans = None
if "hal_score" not in st.session_state: st.session_state.hal_score = 0
if "yt_links" not in st.session_state: st.session_state.yt_links = []
if "is_premium" not in st.session_state: st.session_state.is_premium = False
if "premium_notes" not in st.session_state: st.session_state.premium_notes = None

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.header("Premium Access")
    if not st.session_state.is_premium:
        st.caption("Login to unlock downloadable study guides.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in PREMIUM_ACCOUNTS and PREMIUM_ACCOUNTS[username] == password:
                st.session_state.is_premium = True
                st.success("Welcome to Premium!")
                st.rerun() # Refreshes the app instantly
            else:
                st.error("Invalid credentials.")
    else:
        st.success("Premium Active")
        if st.button("Logout"):
            st.session_state.is_premium = False
            st.session_state.premium_notes = None
            st.rerun()

    st.markdown("---")
    st.header("Settings")
    subject = st.selectbox("Subject", ["All", "OS", "COA", "PPS"])
    learning_level = st.selectbox("Learning Level", ["Beginner", "Intermediate", "Advanced"])
    language = st.selectbox("Explanation Language", ["English", "Hindi", "Bengali", "Telugu", "Tamil"])
    debug = st.checkbox("Show Debug Info")

# --- MAIN ENGINE ---
q = st.text_input("Ask a question about your syllabus:")
if st.button("Search & Analyze", type="primary"):
    if q:
        with st.spinner("Searching textbooks and calculating metrics..."):
            try:
                # 1. THE QUERY REWRITER (Hyper-fast, handles Telugu/Acronyms safely)
                rewrite_prompt = f"You are a search query optimizer. The user asked: '{q}'. Translate this to English if needed, and expand any computer science acronyms or slang. If it is already a clean English phrase, just return the original phrase. Return ONLY the clean, expanded English search phrase, nothing else."
                
                rewrite_res = model.generate_content(
                    rewrite_prompt, 
                    generation_config=genai.GenerationConfig(temperature=0.1)
                )
                
                # SAFE EXTRACTION: Prevents the "finish_reason is 1" crash
                try:
                    smart_q = rewrite_res.text.strip()
                    if not smart_q: 
                        smart_q = q
                except Exception:
                    smart_q = q 
                
                if debug: st.sidebar.write(f"Database Searched For: {smart_q}")

                # 2. SEARCH THE DATABASE WITH THE CLEAN QUERY
                res = get_relevant_course_context(smart_q, subject=subject, level=learning_level)
                
                if not res:
                    st.error(f"Nothing found for {learning_level} level in {subject}.")
                else:
                    st.session_state.data = res
                    ctx = "\n\n".join([f"Source P{r['page']}: {r['text']}" for r in res[:4]])
                    
                    level_prompts = {
                        "Beginner": "Use very simple language, basic analogies, and avoid overly complex jargon.",
                        "Intermediate": "Provide a standard college-level explanation with clear definitions.",
                        "Advanced": "Provide a highly technical, deep dive. Focus on architecture and algorithms."
                    }
                    
                    # 3. THE FINAL GENERATION (Answers in the user's selected language)
                    prompt = f"""
                    You are a strict academic AI. Explain the concept of '{smart_q}' to the user in {language}. 
                    Approach: {level_prompts[learning_level]}
                    CRITICAL: You must construct your explanation ONLY using the facts from the context below. 
                    CONTEXT:\n{ctx}\n
                    Keep core technical terms in English. Use clean Markdown formatting.
                    """
                    
                    ai_res = model.generate_content(
                        prompt, 
                        generation_config=genai.GenerationConfig(temperature=0.1)
                    )
                    st.session_state.ans = ai_res.text
                    st.session_state.hal_score = calculate_hallucination_score(ai_res.text, ctx)
                    st.session_state.yt_links = get_yt_videos(smart_q + " " + subject)
                    st.session_state.premium_notes = None 
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- DISPLAY RESULTS ---
if st.session_state.data and st.session_state.ans:
    c1, c2 = st.columns([1.2, 1]) 
    
    with c1:
        score = st.session_state.hal_score
        color = "#00FF00" if score < 15 else "#FFA500" if score < 40 else "#FF0000"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; padding-right: 10px;"> AI Explanation ({language})</h3>
            <div title="Hallucination Probability: {score}%" 
                 style="width: 15px; height: 15px; border-radius: 50%; background-color: {color}; box-shadow: 0 0 5px {color};">
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(st.session_state.ans)
        
        # --- PREMIUM FEATURE UI ---
        st.markdown("---")
        st.subheader(" Premium Study Guide")
        
        if not st.session_state.is_premium:
            st.info(" **Premium Feature Locked:** Log in via the sidebar to generate and download a comprehensive, multi-page study guide covering this topic, exam questions, and related concepts.")
            st.button("Generate Detailed Notes (.txt)", disabled=True)
        else:
            if not st.session_state.premium_notes:
                if st.button("Generate Detailed Notes"):
                    with st.spinner("Generating premium extensive notes..."):
                        # FIX: Reconstruct context from session state so it exists during this button's rerun
                        current_ctx = "\n\n".join([f"Source P{r['page']}: {r['text']}" for r in st.session_state.data[:4]])
                        
                        notes_prompt = f"""
                        Write a highly detailed 5-part study guide in {language} on '{q}'. 
                        Include: 1) Core Definition, 2) Step-by-Step Mechanism, 3) Real World Use Cases, 4) 3 Likely Exam Questions with Answers, 5) Summary. 
                        Make it highly educational.
                        CRITICAL: Base all facts entirely on this course context:
                        {current_ctx}
                        """
                        notes_res = model.generate_content(notes_prompt)
                        st.session_state.premium_notes = notes_res.text
                        st.rerun()
            
            if st.session_state.premium_notes:
                st.success("Notes Generated Successfully!")
                st.download_button(
                    label=" Download Premium Notes (.txt)",
                    data=st.session_state.premium_notes,
                    file_name=f"{q.replace(' ', '_')}_Notes.txt",
                    mime="text/plain",
                    type="primary"
                )

    with c2:
        st.subheader("Top Reference (Main Definition)")
        data = st.session_state.data
        top_res = data[0]
        st.success(f"Primary Source: {top_res['source']} (Page {top_res['page']})")
        if st.button(f" Open Primary Page {top_res['page']}", key="primary_img"):
            show_page_pop(top_res['source'], top_res['page'])
        with st.expander("View Other Found Pages"):
            for i, item in enumerate(data[1:]):
                st.write(f"**{item['source']}** (Page {item['page']})")
                if st.button(f"Open Page {item['page']}", key=f"b{i}"):
                    show_page_pop(item['source'], item['page'])

        st.subheader("Relevant Tutorials")
        if st.session_state.yt_links:
            for link in st.session_state.yt_links:
                st.video(link)
        else:
            st.caption("No relevant videos found.")