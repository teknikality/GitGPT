import streamlit as st
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uuid

load_dotenv()

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-0125',
    openai_api_key=OPENAI_KEY,
    temperature=0.0,
    streaming=True
)

#ChatPrompt to ensure context is stored and follow up questions are answered . 
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following conversation history and retrieved context to answer the question. 
If addressing a follow-up question, use both the conversation history and new context to provide a coherent response.
If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise.

<conversation_history>
{chat_history}
</conversation_history>

<context>
{context}
</context>

Current question: {question}
"""

prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)

class ChatAgent:
    #Initializes a new ChatAgent instance
    def __init__(self):
        self.history = StreamlitChatMessageHistory(key="chat_history")
        self.embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
        self.index = pc.Index('test-git')
        self.chain = self.setup_chain()
        self.session_id = str(uuid.uuid4())
        self.context_store = {}  

    #Establishes the LangChain pipeline for processing messages.
    def setup_chain(self):
        """Set up a LangChain with message history."""
        return RunnableWithMessageHistory(
            prompt_template | llm,
            lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
    
    #Formats the conversation history into a structured string.
    def format_chat_history(self):
        """Format the chat history for the prompt."""
        formatted_history = ""
        messages = self.history.messages
        for i in range(0, len(messages)-1, 2):  
            if i+1 < len(messages):
                human = messages[i].content if isinstance(messages[i], HumanMessage) else ""
                ai = messages[i+1].content if isinstance(messages[i+1], AIMessage) else ""
                formatted_history += f"Human: {human}\nAssistant: {ai}\n\n"
        return formatted_history
    
    #Generates a context-aware search query based on conversation history.
    def get_enhanced_query(self, question: str) -> str:
        """Generate an enhanced query using chat history context."""
        if len(self.history.messages) < 2:
            return question
        
        # Create a context-aware query
        history_context = self.format_chat_history()
        enhanced_query_prompt = f"""
        Given the following conversation history and new question, create a search query that captures the full context:
        
        History:
        {history_context}
        
        New question: {question}
        
        Search query:"""
        
        response = llm.invoke(enhanced_query_prompt)
        return response.content

    #Retrieves relevant documents from Pinecone using semantic search.
    def retrieve_documents(self, question):
        """Retrieve documents from Pinecone using an enhanced question embedding."""
        # Generate enhanced query using conversation context
        enhanced_query = self.get_enhanced_query(question)
        
        # Generate embedding for the enhanced query
        query_embedding = self.embedding_model.encode(enhanced_query).tolist()
        query_embedding = [float(x) for x in query_embedding]
        
        try:
            search_results = self.index.query(vector=query_embedding, top_k=3, include_metadata=True)
        except Exception as e:
            print("Error querying Pinecone:", str(e))
            raise
            
        filtered_results = [
            match for match in search_results["matches"]
            if match["score"] >= 0.3
        ]
        
        # Store context for this question
        self.context_store[question] = filtered_results
        
        return filtered_results

    #Formats retrieved documents for inclusion in the prompt.
    def format_retrieved_content_for_prompt(self, search_results):
        """Format the retrieved content for the prompt."""
        context = ""
        if not search_results:
            return "No relevant sources found for this query."
            
        for match in search_results:
            title = match["metadata"]["title"]
            url = match["metadata"]["url"]
            text = match["metadata"]["text"]
            context += f"Title: {title}\nURL: {url}\nText: {text}\n\n"
        return context

    #Generates an answer using the LLM
    def get_answer_from_llm(self, context, question):
        """Generate answer using the LLM with given context and question."""
        chat_history = self.format_chat_history()
        
        response = self.chain.invoke(
            {
                "context": context,
                "question": question,
                "chat_history": chat_history
            },
            {
                "configurable": {
                    "session_id": self.session_id
                }
            }
        )
        return response.content

    def display_messages(self):
        """Display chat messages."""
        if len(self.history.messages) == 0:
            self.history.add_ai_message("Welcome")
        for msg in self.history.messages:
            st.chat_message(msg.type).write(msg.content)

    def get_user_question(self) -> str:
        """Get user input from chat input field."""
        return st.chat_input(placeholder="Ask me anything!")

    def display_user_question(self, question: str):
        """Display the user's question in the chat."""
        st.chat_message("human").write(question)

    def display_ai_response(self, response: str):
        """Display the AI's response in the chat."""
        st.chat_message("ai").write(response)

    def display_sources(self, search_results):
        """Display the sources retrieved with clickable links."""
        st.write("#### Sources:")
        match = search_results
        title = match["metadata"]["title"]
        url = match["metadata"]["url"]
        st.write(f"**[{title}]({url})**")