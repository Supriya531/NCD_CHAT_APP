import streamlit as st
# from utils.asr import Transcription, transcribe_audio_incredibly_fast_whisper, transcribe_audio_whisper_diarization
# from utils.translation import TranslationIndic
# from utils.llm import get_groq_response, get_cohere_response, get_anthropic_response, get_replicate_response
# from utils.tts import synthesize_speech
from translation import TranslationIndic
from llm import run_qa
import os
import numpy as np
from streamlit_mic_recorder import mic_recorder
import time
# Import SPRING_INX_Transcripts from infer.py
# from utils.SPRING_INX_ASR.infer1 import SPRING_INX_Transcripts
translator_indic = TranslationIndic()
# Function to translate roles between llm model and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

def main():
    st.title("let's chat on NCD..")

    # Language selection in the sidebar
    languages = {
        "English": "eng",
        # "Bengali": "ben",
        # "Gujarati": "guj",
        # "Hindi": "hin",
        # "Kannada": "kan",
        # "Malayalam": "mal",
        # "Marathi": "mar",
        # "Odia": "ory",
        # "Punjabi, Eastern": "pan",
        # "Tamil": "tam",
        "Telugu": "tel",
    }

    language_display = st.sidebar.selectbox("Select Language", list(languages.keys()))
    language_code = languages[language_display]

    # # ASR model selection in the sidebar (only for English)
    # if language_code == "eng":
    #     asr_model_options = ["Whisper", "Incredibly Fast Whisper", "Whisper Diarization"]
    #     asr_model = st.sidebar.selectbox("Select ASR Model", asr_model_options)
    # else:
    #     asr_model = None

    # Mapping language codes from app.py to asr_indic.py
    language_code_mapping = {
        "eng": "eng",
        "ben": "ben_Beng",
        "guj": "guj_Gujr",
        "hin": "hin_Deva",
        "kan": "kan_Knda",
        "mal": "mal_Mlym",
        "mar": "mar_Deva",
        "ory": "ory_Orya",
        "pan": "pan_Guru",
        "tam": "tam_Taml",
        "tel": "tel_Telu",
    }

    mapped_language_code = language_code_mapping.get(language_code, language_code)

    # LLM model selection in the sidebar
    llm_model_display_names = [
         "Llama3-70b-8192", "Llama3-8b-8192"
    ]
    llm_model_names = [name.lower() for name in llm_model_display_names]
    llm_model_display_to_internal = dict(zip(llm_model_display_names, llm_model_names))
    llm_model_display = st.sidebar.selectbox("Select LLM Model", llm_model_display_names)

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = {
            "history": []
        }

    # Display the chat history
    for message in st.session_state.chat_session["history"]:
        with st.chat_message(translate_role_for_streamlit(message["role"])):
            st.markdown(message["content"])


    # Text input for user's message at the bottom of the page
    user_message = st.chat_input("Ask LLM model")
    if user_message:
        # Add user's message to chat and display it
        st.session_state.chat_session["history"].append({"role": "user", "content": user_message})
        st.chat_message("user").markdown(user_message)

        # Send user's message to LLM model and get the response
        response = process_text_message(user_message, mapped_language_code, llm_model_display_to_internal[llm_model_display])

        # Display LLM response
        st.session_state.chat_session["history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        # Rerun the script to ensure chat history updates immediately
        st.rerun()

    
def process_text_message(message, language, llm_model):
    start_time = time.time()
    
    # Get chat history from session state
    chat_history = st.session_state.chat_session["history"]

    # Add user's message to chat history
    # chat_history.append({"role": "user", "content": message})
    
    # Prepare the messages format for the LLM
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]

    if language == "eng":
        if llm_model in ["gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]:
            # response = get_groq_response(messages, llm_model)
            response = run_qa(messages, llm_model)
        elif llm_model in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]:
            response = get_anthropic_response(message, llm_model)
        elif llm_model in ["gemini-1.0-pro"]:
            response = get_gemini_response(message, llm_model)
        elif llm_model.startswith("llama-2"):
            response = get_replicate_response(message, llm_model)
        else:
            response = get_cohere_response(message, llm_model)
        
        return response
    else:
        translated_message = translator_indic.translate_to_english(message, "tel_Telu")
        

        if llm_model in ["llama3-70b-8192", "llama3-8b-8192"]:
            response = run_qa(translated_message, llm_model)
        elif llm_model in ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]:
            response = get_anthropic_response(translated_transcript, llm_model)
        elif llm_model.startswith("llama-2"):
            response = get_replicate_response(translated_transcript, llm_model)
        else:
            response = get_cohere_response(translated_transcript, llm_model)
        
        translated_response = translator_indic.translate_to_indic(response, language)
        print(f"{llm_model} response in Indic: {translated_response}")

        # Add assistant's response to chat history
        chat_history.append({"role": "assistant", "content": translated_response})
        
        # Save updated chat history to session state
        st.session_state.chat_session["history"] = chat_history
        
        return translated_response

if __name__ == "__main__":
    # transcriber_indic = TranscriptionIndic()
    # translator_indic = TranslationIndic()
    main()