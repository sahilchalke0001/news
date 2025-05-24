import streamlit as st
from newspaper import Article
from transformers import pipeline
import nltk
import requests
import re
import tempfile
import math

# Attempt to import moviepy and handle potential errors
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="News Article Summarizer & Reel Generator", page_icon="📰", layout="wide")

# --- NLTK Setup ---
# Download the 'punkt' tokenizer model if not already present.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:  # <-- Change to this
    nltk.download('punkt')

# --- Hugging Face Model Setup ---
# Initialize the summarization pipeline.
@st.cache_resource # Cache the model
def load_summarizer_model():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        st.info("Ensure an active internet connection to download the model, or check model path.")
        return None

summarizer = load_summarizer_model()

# --- DeepAI Image Generation Function ---
def generate_image_from_text(prompt_text):
    """Generates an image using the DeepAI text2img API."""
    st.info("Generating image based on summary using DeepAI...")

    # --- IMPORTANT: REPLACE WITH YOUR KEY ---
    deepai_api_key = "YOUR_DEEPAI_API_KEY_HERE"
    # --- IMPORTANT: REPLACE WITH YOUR KEY ---

    if not deepai_api_key or deepai_api_key == "YOUR_DEEPAI_API_KEY_HERE":
        st.error("DeepAI API key is missing. Please get one from deepai.org and replace 'YOUR_DEEPAI_API_KEY_HERE' in the code.")
        return None

    api_url = "https://api.deepai.org/api/text2img"
    headers = {'api-key': deepai_api_key}
    payload = {'text': prompt_text}

    try:
        response = requests.post(api_url, headers=headers, data=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get("output_url"):
            return result["output_url"]
        elif result.get("err"):
            st.error(f"DeepAI API error: {result['err']}")
            return None
        else:
            st.error(f"Failed to generate image: Unexpected DeepAI response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"DeepAI API call failed: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during image generation: {e}")
        return None

# --- Video Generation Function ---
def generate_video_reel(summary_text, image_url, title):
    """Generates a video reel with text on a white background and adds a download button."""
    if not MOVIEPY_AVAILABLE:
        st.error("Video generation requires the 'moviepy' library. Please install it (`pip install moviepy`).")
        st.warning("Note: MoviePy also requires FFmpeg. Ensure it's installed and accessible in your system's PATH.")
        return

    st.info("Generating video reel... This might take some time.")
    with st.spinner("Processing video..."):
        try:
            words_per_second = 3
            word_count = len(summary_text.split())
            duration = math.ceil(word_count / words_per_second) + 3 # Add buffer time

            video_width = 800
            video_height = 600

            # Create a white background clip
            background = mp.ColorClip(size=(video_width, video_height), color=(255, 255, 255), duration=duration)

            # Create a TextClip for the summary
            # We use 'caption' method for basic word wrapping.
            # Adjust fontsize, font, and size for better results.
            text_clip = mp.TextClip(
                summary_text,
                fontsize=30,
                color='black',
                bg_color='white',
                size=(video_width * 0.85, video_height * 0.8), # Use 85% width, 80% height
                method='caption',
                align='West'
            ).set_duration(duration).set_position('center')

            # Create the final video by overlaying text on background
            final_video = mp.CompositeVideoClip([background, text_clip])

            # Write the video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                video_path = tmpfile.name
                # Use 'ultrafast' preset for speed, but lower quality.
                # Consider 'medium' or 'slow' for better quality if performance allows.
                # 'threads' can speed up processing.
                # Ensure 'libx264' codec for MP4.
                final_video.write_videofile(
                    video_path,
                    fps=24,
                    codec="libx264",
                    preset="ultrafast",
                    threads=4,
                    logger=None # Suppress verbose output
                )

            st.success("Video reel generated!")
            st.video(video_path) # Display the video in Streamlit

            # Provide the download button
            with open(video_path, "rb") as file:
                st.download_button(
                    label="Download Video Reel (.mp4)",
                    data=file,
                    file_name=f"{title}_reel.mp4",
                    mime="video/mp4"
                )

        except NameError: # Catch if moviepy wasn't imported
             st.error("MoviePy library not found. Cannot generate video.")
        except Exception as e:
            st.error(f"Failed to generate video: {e}")
            st.warning("Ensure 'moviepy' and its dependency 'FFmpeg' are correctly installed and accessible. Text rendering might also require 'ImageMagick'.")

# --- Streamlit UI ---
st.title("📰 News Article Summarizer & Reel Generator")
st.markdown("---")
st.write("Paste a news article URL to get a summary, image, audio, and a simple video reel.")

url_input = st.text_input("Enter Article URL:")

# Initialize session state
if 'summary_text' not in st.session_state:
    st.session_state.summary_text = ""
if 'generated_image_url' not in st.session_state:
    st.session_state.generated_image_url = ""
if 'article_title' not in st.session_state:
    st.session_state.article_title = ""

# --- Summarize Button Logic ---
if st.button("Summarize Article"):
    if url_input:
        try:
            with st.spinner("Fetching and summarizing article..."):
                article = Article(url_input)
                article.download()
                article.parse()

                if not article.text:
                    st.warning("Could not extract content. Check URL, paywalls, or website structure.")
                    st.session_state.summary_text = ""
                    st.session_state.article_title = ""
                else:
                    st.session_state.article_title = article.title if article.title else 'No_title_found'
                    st.subheader("Original Article:")
                    st.markdown(f"**Title:** {st.session_state.article_title}")
                    st.markdown(f"**Authors:** {', '.join(article.authors) if article.authors else 'N/A'}")
                    st.markdown(f"**Published:** {article.publish_date.strftime('%Y-%m-%d') if article.publish_date else 'N/A'}")
                    st.markdown(f"[Read Original]({url_input})")
                    st.text_area("Original Text (Snippet)", article.text[:1000] + "...", height=150)

                    sentences = nltk.sent_tokenize(article.text)
                    processed_text = " ".join(sentences)

                    if summarizer:
                        summary_result = summarizer(processed_text, max_length=150, min_length=40, do_sample=False)
                        st.session_state.summary_text = summary_result[0]['summary_text']
                        st.subheader("Generated Summary:")
                        st.success(st.session_state.summary_text)
                    else:
                        st.error("Summarizer not loaded.")
                        st.session_state.summary_text = ""

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            st.warning("Ensure URL is valid. Some sites block scraping.")
            st.session_state.summary_text = ""
            st.session_state.article_title = ""
    else:
        st.warning("Please enter a URL.")

# --- Reel Generation Section ---
if st.session_state.summary_text:
    st.markdown("---")
    st.header("✨ Generate Reel from Summary")
    st.write("Create an image, an audio summary, and a video reel.")

    if st.button("Generate Image, Audio & Video Reel"):
        image_prompt = f"A high-quality news illustration representing: {st.session_state.summary_text[:120]}..."
        st.session_state.generated_image_url = generate_image_from_text(image_prompt)

        if st.session_state.generated_image_url:
            st.success("Image generated successfully!")
            st.image(st.session_state.generated_image_url, caption="Generated Image", use_column_width=True)

            # Sanitize title for filenames
            sanitized_title = re.sub(r'[^\w\s-]', '', st.session_state.article_title).strip().replace(' ', '_')[:50]

            # --- Video Generation Call ---
            generate_video_reel(st.session_state.summary_text, st.session_state.generated_image_url, sanitized_title)

            # --- Audio Generation (JavaScript) ---
            st.subheader("🔊 Audio Summary & Downloads")
            escaped_summary = st.session_state.summary_text.replace('`', '\\`').replace('\n', ' ')
            escaped_image_url = st.session_state.generated_image_url.replace('`', '\\`')

            js_code = f"""
            <script>
            const summaryText = `{escaped_summary}`;
            const imageUrl = `{escaped_image_url}`;
            const articleTitle = `{sanitized_title}`;

            function speakSummary() {{
                if ('speechSynthesis' in window) {{
                    window.speechSynthesis.cancel(); // Cancel any previous speech
                    const utterance = new SpeechSynthesisUtterance(summaryText);
                    utterance.lang = 'en-US';
                    window.speechSynthesis.speak(utterance);
                }} else {{
                    alert("Text-to-speech not supported in this browser.");
                }}
            }}

            function downloadImage() {{
                const a = document.createElement('a');
                a.href = imageUrl;
                a.download = `summary_image_${articleTitle}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }}

            // Add buttons if they don't exist
            let controlsDiv = document.getElementById('reel-controls');
            if (!controlsDiv) {{
                controlsDiv = document.createElement('div');
                controlsDiv.id = 'reel-controls';
                document.body.appendChild(controlsDiv); // Or append to a specific Streamlit container
            }}

            // Clear existing buttons before adding new ones
            controlsDiv.innerHTML = `
                <button onclick="speakSummary()" style="padding: 10px 15px; margin: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    ▶️ Play Audio Summary
                </button>
                <button onclick="downloadImage()" style="padding: 10px 15px; margin: 5px; background-color: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    🖼️ Download Image
                </button>
                <p><small>Audio download via browser's text-to-speech is complex and not implemented here.</small></p>
            `;

            // Ensure the div is visible (may need better integration with Streamlit's layout)
            const stApp = window.parent.document.querySelector('.main .block-container');
            if (stApp && !document.getElementById('reel-controls')) {{
                stApp.appendChild(controlsDiv);
            }}

            </script>
            <div id="reel-controls-placeholder"></div>
            """
            st.components.v1.html(js_code, height=100)
        else:
            st.error("Could not generate image, so reel generation is skipped.")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.info(
    "This app summarizes news articles, generates a relevant image via DeepAI, "
    "provides text-to-speech audio, and attempts to create a simple video reel "
    "using text-on-white background."
    "\n\n**Key Libraries & APIs Used:**"
    "\n- `streamlit`: Web interface."
    "\n- `newspaper3k`: Article scraping."
    "\n- `nltk`: Sentence tokenization."
    "\n- `transformers`: Text summarization."
    "\n- `requests`: API calls."
    "\n- `DeepAI API`: Image generation."
    "\n- `moviepy` (Optional): Video generation."
    "\n- `Web Speech API`: Text-to-speech."
)
st.sidebar.warning(
    "**Note on Video:** Video generation requires `moviepy` and `FFmpeg`. "
    "These must be installed separately. FFmpeg needs to be in your system's PATH. "
    "Generating videos can be slow and resource-intensive."
)
st.sidebar.error("**API Key:** Remember to replace `'YOUR_DEEPAI_API_KEY_HERE'` in the code with your actual DeepAI key.")
st.sidebar.markdown("[View on GitHub (Placeholder)](https://github.com/your-repo)")
