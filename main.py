import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import BartForConditionalGeneration, BartTokenizer
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm


st.set_page_config(page_title="Youtube Transcript Summarizer", layout="wide")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("dis_img.png", width=300)

with col3:
    st.write(' ')

def extract_video_id(link):
    """
    Extract the video id from a YouTube video link.
    """
    # Parse the link using urlparse
    parsed_url = urlparse(link)
    
    if parsed_url.netloc == "www.youtube.com":
        # Extract the video id from the query parameters for the www.youtube.com format
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        else:
            return None
    elif parsed_url.netloc == "youtu.be":
        # Extract the video id from the path for the youtu.be format
        path = parsed_url.path
        if path.startswith("/"):
            path = path[1:]
        return path
    else:
        # Return None for all other link formats
        return None

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


def GetSubtitles(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


def clearing_text(transcript):
    new = ""

    for i in transcript:
        new += i["text"] + ". "
    return new

st.title('Youtube Summary Generator')
st.subheader(
    'Summarize any video with captions.')

video_link = st.text_area("Enter Youtube Link 😁" ,height=1)

video_id = extract_video_id(video_link)


if st.button('Generate Summary'):
    with st.spinner('Getting Subtitles'):
        transcript = GetSubtitles(video_id)
        display = clearing_text(transcript)
        print(display)
        st.success('Done !')
    with st.spinner('Generating Summary.....'):
        # Define a function to summarize a segment of text
        def summarize_segment(segment, max_length=200):
            inputs = tokenizer.encode(segment, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=max_length, min_length=int(max_length / 5), length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        # Define a function to split the text into segments and summarize each segment
        def summarize_text_segments(text, max_length_per_segment=500, max_segments=15):
            segments = []
            for i in range(0, len(text), max_length_per_segment):
                segment = text[i:i+max_length_per_segment]
                segments.append(segment)
    
            summaries = []
            for segment in tqdm(segments[:max_segments], desc="Summarizing", unit="segment"):
                summary = summarize_segment(segment)
                summaries.append(summary)
            
            final_summary = ' '.join(summaries)
            return final_summary

            # Concatenate all transcript text into a single string
        text = " ".join([i["text"] for i in transcript])

            # Generate the summary for text segments
        final_summary = summarize_text_segments(text)


        row1, row2 = st.columns(2)

        with row1:
            st.text_area("Original", display, height=600)

        with row2:
            st.text_area("Summary", final_summary, height=600)

        #st.text_area("Analysis", final_summary,height=1000)
        st.success('Done!')

