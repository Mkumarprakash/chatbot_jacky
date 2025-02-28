
import os 
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

#Addition for voice chat
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play



load_dotenv()


groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)
embeddings = OllamaEmbeddings(model='gemma2:2b')
mydb = Chroma(persist_directory="./chroma_db",embedding_function=embeddings)
retriever = mydb.as_retriever(search_type='similarity',search_kwargs={"k":6})

st.title("Welcome to Prakash's CHATBOT,🎙️ Prakash's Voice-Enabled Chatbot")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
query = st.chat_input("Ask me anything")

AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

# Speech Recognition Function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # Capture voice input
            text = recognizer.recognize_google(audio)  # Convert speech to text
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Speech Recognition service is unavailable."

# Text-to-Speech Function
def speak(text):
    tts = gTTS(text=text, lang="en")
    
    # Save to BytesIO
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    # Load audio and play
    audio = AudioSegment.from_file(audio_bytes, format="mp3")
    play(audio)


system_prompt = (
    "You are an assistant for question answering all questions from prakash to help him to improve his knowledge. and your name is jacky and also give that much answer how much required not morethan that extra word . and also highlight the specific words "
    "Use the following pieces of retrieved context to answer the question. and read fast and don't  read emoji and  don't my name again and again "
    "Make sure you talk very polite with the customer and  write anything good or bad  which is appropriate."
    "Your tone of reply should always be exciting and luring to me."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")
])

for user_query, bot_response in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(bot_response)



# Capture voice input
if st.button("🎙️ Start Voice Input"):
    query = recognize_speech()
    st.write(f"**You:** {query}")
else:
    query = st.text_input("💬 Type your question here")

if query:
    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Create the response chain
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get response
    response = rag_chain.invoke({'input': query})
    bot_reply = response['answer']

    # Display chatbot reply
    with st.chat_message("assistant"):
        st.write(bot_reply)
    # Speak the response
    speak(bot_reply)

    # Store the interaction in chat history
    st.session_state.chat_history.append((query, bot_reply))


