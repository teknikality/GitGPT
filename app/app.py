import os
import streamlit as st
from components.chat_utils import ChatAgent
import deepl
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
class MultilingualChatBot:
    def __init__(self):
        DEEPL_API_KEY=os.environ.get("DEEPL_API_KEY")
        self.translator = deepl.Translator(DEEPL_API_KEY)
        
        self.supported_languages = {
            "English": "EN-US",  
            "French": "FR"       
        }
        self.target_lang_mapping = {
            "English": "en",     
            "French": "fr"       
        }
        
    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text to target language if needed."""
        if not text:
            return text
            
        current_lang = st.session_state.get("selected_language", "English")
        if current_lang == "English":
            return text
            
        try:
            # Use the correct lowercase language code for the target language
            result = self.translator.translate_text(
                text, 
                target_lang=self.target_lang_mapping[current_lang]
            )
            return str(result)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return text
            
    def get_translated_content(self, content: Dict[str, str]) -> Dict[str, str]:
        """Translate all content in the dictionary to selected language."""
        current_lang = st.session_state.get("selected_language", "English")
        if current_lang == "English":
            return content
            
        translated_content = {}
        target_lang = self.target_lang_mapping[current_lang]
        
        for key, value in content.items():
            translated_content[key] = self.translate_text(value, target_lang)
            
        return translated_content

def main():
    # Initialize multilingual support
    if "multilingual_bot" not in st.session_state:
        st.session_state.multilingual_bot = MultilingualChatBot()
    
    # Language selector
    st.sidebar.selectbox(
        "Select Language / Sélectionner la langue",
        options=list(st.session_state.multilingual_bot.supported_languages.keys()),
        key="selected_language"
    )
    
    # Translate title and subheader based on selected language
    title = st.session_state.multilingual_bot.translate_text(
        "Bharath's Attempt",
        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
    )
    subheader = st.session_state.multilingual_bot.translate_text(
        "Basic RAG Chatbot",
        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
    )
    description = st.session_state.multilingual_bot.translate_text(
        "Implemented a basic RAG chatbot to answer queries based on GITLAB docs",
        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
    )
    
    st.title(title)
    st.subheader(subheader)
    st.write(description)

    # Initialize the ChatAgent if not already done
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = ChatAgent()

    # Define example prompts and translate them
    example_prompts = [
         
        "How can I contribute to GitLab?",
        "Explain GitLab's CI/CD features.",
        "What are the best practices for using GitLab?",
        "Tell me about GitLab's security features."
        
    ]
    
    translated_prompts = [
        st.session_state.multilingual_bot.translate_text(
            prompt,
            st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
        ) for prompt in example_prompts
    ]

    # Function to populate the input field with an example prompt
    def set_example_prompt(prompt: str):
        st.session_state["example_question"] = prompt

    # Display translated example prompts as buttons
    st.write(st.session_state.multilingual_bot.translate_text(
        "#### Try out these example questions:",
        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
    ))
    
    for prompt in translated_prompts:
        if st.button(prompt):
            set_example_prompt(prompt)

    def start_conversation():
        """Start the chat conversation with translation support."""
        # Display initial messages
        st.session_state.chat_agent.display_messages()

        # Get user input, using example question if set
        user_question = st.session_state.chat_agent.get_user_question() or st.session_state.get("example_question", "")
        
        if user_question:
            st.session_state["example_question"] = ""  # Clear the example question after use
            
            # Translate user question to English for processing
            if st.session_state.selected_language != "English":
                processing_question = st.session_state.multilingual_bot.translate_text(
                    user_question,
                    "en"  # Always translate to English for processing
                )
            else:
                processing_question = user_question
                
            st.session_state.chat_agent.display_user_question(user_question)
            
            with st.spinner(st.session_state.multilingual_bot.translate_text(
                "Generating response...",
                st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
            )):
                # Retrieve relevant documents from Pinecone
                search_results = st.session_state.chat_agent.retrieve_documents(processing_question)
                context = st.session_state.chat_agent.format_retrieved_content_for_prompt(search_results)

                if context == "No relevant sources found for this query.":
                    no_results_message = st.session_state.multilingual_bot.translate_text(
                        "No relevant sources found for this query.",
                        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
                    )
                    st.session_state.chat_agent.display_ai_response(no_results_message)
                    st.session_state.chat_agent.history.add_user_message(user_question)
                    st.session_state.chat_agent.history.add_ai_message(no_results_message)
                else:
                    answer = st.session_state.chat_agent.get_answer_from_llm(context, processing_question)
                    # Translate answer to selected language
                    translated_answer = st.session_state.multilingual_bot.translate_text(
                        answer,
                        st.session_state.multilingual_bot.target_lang_mapping[st.session_state.selected_language]
                    )
                    st.session_state.chat_agent.display_ai_response(translated_answer)
                    st.session_state.chat_agent.display_sources(search_results[0])

    # Start conversation
    start_conversation()

if __name__ == "__main__":
    main()