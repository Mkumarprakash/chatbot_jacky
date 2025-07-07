import os 
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

# Addition for voice chat
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import time

# Load environment variables
load_dotenv()

# Set ffmpeg path for pydub
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

# Load model and retriever
groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
embeddings = OllamaEmbeddings(model='gemma2:2b')
mydb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = mydb.as_retriever(search_type='similarity', search_kwargs={"k": 6})

# Streamlit UI setup
st.title("Welcome to Prakash's CHATBOT 🎙️ - Voice Enabled Assistant")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

# Prompt template
system_prompt = (
    "You are an assistant for question answering all questions from who ask to help him/her to improve his knowledge. "
    "Before that you ask his/her name for remember. Your name is Jacky. Keep answers concise—don't use extra words. "
    "Highlight specific keywords.\n\n"
    "Use the following pieces of retrieved context to answer the question. Do not read emojis. "
    "Avoid repeating the user's name. Speak politely and constructively. Your tone should be exciting and luring.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Speech Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Speech Recognition service is unavailable."

# Text-to-Speech
def speak(text, speed=1.3):
    tts = gTTS(text=text, lang="en")
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    audio = AudioSegment.from_file(audio_bytes, format="mp3")
    faster_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)
    play(faster_audio)

# Chat display history
for user_query, bot_response in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(bot_response)

# Live voice toggle buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎙️ Start Voice Chat"):
        st.session_state.voice_active = True
with col2:
    if st.button("🛑 Stop Voice Chat"):
        st.session_state.voice_active = False

# Live voice chat loop
if st.session_state.voice_active:
    st.info("🟢 Voice chat is active... Speak your question.")

    while st.session_state.voice_active:
        query = recognize_speech()

        if query:
            with st.chat_message("user"):
                st.write(query)

            # Create the RAG chain
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Get response
            response = rag_chain.invoke({'input': query})
            bot_reply = response['answer']

            with st.chat_message("assistant"):
                st.write(bot_reply)

            # Speak and store
            speak(bot_reply)
            st.session_state.chat_history.append((query, bot_reply))

        time.sleep(1)  # prevent tight loop

# Manual input fallback
if not st.session_state.voice_active:
    query = st.chat_input("💬 Type your question here")

    if query:
        with st.chat_message("user"):
            st.write(query)

        question_answer_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({'input': query})
        bot_reply = response['answer']

        with st.chat_message("assistant"):
            st.write(bot_reply)

        speak(bot_reply)
        st.session_state.chat_history.append((query, bot_reply))
