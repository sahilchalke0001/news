import streamlit as st
from newspaper import Article
from transformers import pipeline
import nltk
import requests
import re
import torch

# First install required version: pip install googletrans==3.1.0a0
from googletrans import Translator

# Set page configuration
st.set_page_config(page_title="News Article Summarizer & Translator", page_icon="📰", layout="wide")

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt')

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.sidebar.info("Using CPU - slower performance")

# --- Model Loading ---
@st.cache_resource
def load_summarizer_model(device):
    try:
        model = pipeline("summarization",
                         model="sshleifer/distilbart-cnn-12-6",
                         device=0 if device == "cuda" else -1)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

summarizer = load_summarizer_model(DEVICE)

# --- Translator Setup ---
@st.cache_resource
def get_translator():
    try:
        return Translator()
    except Exception as e:
        st.error(f"Translator init error: {e}")
        return None

translator = get_translator()

# --- Translation Function ---
def translate_text(text, target_lang):
    if not translator:
        return "Translation unavailable"
    
    try:
        # The googletrans library's translate method is synchronous in 3.1.0a0
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as e:
        st.error(f"Translation error ({target_lang}): {e}")
        return "Translation failed"

# --- UI Components ---
st.title("📰 News Article Summarizer & Translator")
st.markdown("---")

# Session state management
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'trans_hindi' not in st.session_state:
    st.session_state.trans_hindi = ""
if 'trans_marathi' not in st.session_state:
    st.session_state.trans_marathi = ""
if 'article_image' not in st.session_state:
    st.session_state.article_image = None


# URL input and processing
url = st.text_input("Enter News Article URL:")
if st.button("Process Article"):
    if url:
        with st.spinner("Analyzing article..."):
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                if not article.text:
                    st.warning("No content found")
                else:
                    # Display article metadata
                    st.subheader("Original Content")
                    st.write(f"**Title:** {article.title}")
                    
                    # --- Display Image ---
                    if article.top_image:
                        st.session_state.article_image = article.top_image
                        # CORRECTED LINE: using use_container_width instead of use_column_width
                        st.image(st.session_state.article_image, caption=article.title, use_container_width=True)
                    elif article.images:
                        # Fallback to the first image if top_image is not found
                        first_image = list(article.images)[0]
                        st.session_state.article_image = first_image
                        # CORRECTED LINE: using use_container_width instead of use_column_width
                        st.image(st.session_state.article_image, caption=article.title, use_container_width=True)
                    else:
                        st.session_state.article_image = None
                        st.info("No main image found for this article.")

                    st.write(f"**Published:** {article.publish_date or 'N/A'}")
                    st.write(f"**Authors:** {', '.join(article.authors) or 'N/A'}")
                    
                    # Generate summary
                    if summarizer:
                        summary = summarizer(article.text,
                                             max_length=150,
                                             min_length=50,
                                             do_sample=False)[0]['summary_text']
                        st.session_state.summary = summary
                        
                        # Generate translations
                        # Using a separate spinner for translations as they can take time
                        with st.spinner("Translating summary..."):
                            st.session_state.trans_hindi = translate_text(summary, 'hi')
                            st.session_state.trans_marathi = translate_text(summary, 'mr')
                        
                        # Display results
                        st.subheader("English Summary")
                        st.success(st.session_state.summary)
                        
                        st.subheader("Translations")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Hindi**")
                            st.info(st.session_state.trans_hindi)
                        with col2:
                            st.markdown("**Marathi**")
                            st.info(st.session_state.trans_marathi)
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.session_state.article_image = None # Clear image on error
    else:
        st.warning("Please enter a valid URL")

# --- Audio Controls ---
# This section will appear if a summary is available, providing playback controls.

# --- Audio Section ---
if st.session_state.summary:
    st.markdown("---")
    st.subheader("Audio Summary")
    # Prepare the summary text for JavaScript *before* the f-string
    # Escape backslashes first, then double quotes, then single quotes
    js_safe_summary = st.session_state.summary.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    
    audio_html = f"""
    <script>
    function speak() {{
        var msg = new SpeechSynthesisUtterance();
        msg.text = "{js_safe_summary}";
        window.speechSynthesis.speak(msg);
    }}

    function pauseSummary() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.pause(); // Pause ongoing speech
        }}
    }}

    function stopSummary() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel(); // Stop and clear all speech
        }}
    }}
    </script>
    <button onclick="speak()" style="padding: 10px 20px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px;">
        ▶️ Play Summary
    </button>
    <button onclick="stopSummary()" style="padding: 10px 20px; margin: 5px; background-color: #F44336; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px;">
        ⏹️ Stop Summary
    </button>
    """
    st.components.v1.html(audio_html, height=80) # Increased height to accommodate buttons

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""This application uses:
- Hugging Face Transformers for summarization
- Newspaper3k for article extraction
- Googletrans v3.1.0 for translations
- Web Speech API for audio playback""")
st.sidebar.warning("""Note: Translation reliability depends on 
Google Translate's web interface stability""")