import streamlit as st
import io
import re
import textwrap
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from transformers import pipeline
from fpdf import FPDF 
@st.cache_data(ttl=4)
def get_transcript(video_id: str,lang:str='en'):
    try:
        transcript=YouTubeTranscriptApi.get_transcript(video_id,languages=[lang])
    except TranscriptsDisabled:
        st.error("transcript is not provided for the video")
    except Exception as e:
        st.error(f"Could not fetch the transcripts because of{e}")
    #joining raw text
    raw=" ".join([item["text"]for item in transcript])
    cleaned=re.sub(r"\[.*?\]","",raw)
    cleaned=re.sub(r"\s+"," ",cleaned)
    return cleaned
@st.cache_resource(ttl=2)
def summarizer():
    # laoding a summarizer model
    return pipeline("Summarization",model="facebook/bart-large-cnn",device_map="auto")
def break_text(text: str,maxtokens: int =1024):
    wrapped=textwrap.TextWrapper(width=maxtokens,break_long_words=False,replace_whitespace=False)
    for chunk in wrapped.wrap(text):
        yield chunk
def summarize(text:str):
    summary_gen=summarizer()
    pieces=break_text(text)
    summaries=[summarizer(chunk,max_length=150,min_length=20,do_sample=False)[0]["summary text"] for chunk in pieces]
    return " ".join(summaries)
def pdf_from_text(text:str)->io.BytesIO:
    pdf=FPDF()
    pdf.set_auto_page_break(auto=True,margin=15)
    pdf.add_page()
    pdf.set_font("Times",size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0,10,line)
    buff=io.BytesIO()
    pdf.output(buff)
    buff.seek(0)
    return buff


# Streamlit UI

st.set_page_config(page_title="YouTube summarizer", page_icon="🎥",layout="wide")
st.title("Youtube video summarizer 🎥")
url=st.text_input("Enter your youtube video url link",placeholder="https://youtube.video/example")
if st.button("Generate summary",type="primary"):
    if not url:
        st.error("Please give out a valid url")
    else:
        video_id_match=re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if not video_id_match:
            st.error("Could not extract video ID from the url")
        else:
            vid=video_id_match.group(1)
            with st.spinner("Fetching transcript"):
                transcript=get_transcript(vid)
            if transcript:
                with st.spinner("Summarizing..."):
                    summary=summarize(transcript)
                st.subheader("Summary")
                st.write(summary)

                st.download_button("⬇️ Download .pdf",
                data=pdf_from_text(summary),
                file_name=f"{vid}_summary.pdf",
                mime="application/pdf")
st.markdown(
   """ Built with [**youtube‑transcript‑api**](https://pypi.org/project/youtube-transcript-api/), 🤗 **Transformers**, and **Streamlit**."""
)
