import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class HealthcareChatbot:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.vectorizer = None
        self.description_vectors = None
        self.load_data()
        self.train_model()
    
    def load_data(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.dataset_path)
        
        # Clean the data
        self.df = self.df.dropna()
        self.df['Description'] = self.df['Description'].str.lower()
        self.df['Patient'] = self.df['Patient'].str.lower()
        self.df['Doctor'] = self.df['Doctor'].str.lower()
    
    def train_model(self):
        """Train the similarity model"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.description_vectors = self.vectorizer.fit_transform(self.df['Description'])
    
    def get_response(self, user_input):
        """Get concise bot response based on user input"""
        user_input_clean = user_input.lower()
        
        # Vectorize user input
        user_vector = self.vectorizer.transform([user_input_clean])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vector, self.description_vectors)
        best_match_idx = similarities.argmax()
        
        if similarities[0][best_match_idx] > 0.1:  # Similarity threshold
            best_match = self.df.iloc[best_match_idx]
            response = f"Based on your symptoms, this might be related to: {best_match['Description']}. Recommended specialist: {best_match['Doctor']}."
        else:
            response = "I understand your health concern. For accurate diagnosis, please consult with a healthcare professional."
        
        return response

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'bot' not in st.session_state:
        try:
            st.session_state.bot = HealthcareChatbot('ai-medical-chatbot.csv')
            st.session_state.bot_initialized = True
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.session_state.bot_initialized = False
    if 'last_input' not in st.session_state:
        st.session_state.last_input = ""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="HealthAssist AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for clean interface
    st.markdown("""
    <style>
    .chat-message {
        padding: 12px 16px;
        border-radius: 15px;
        margin: 8px 0;
        line-height: 1.4;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 60px;
        border: 1px solid #90caf9;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 60px;
        border-left: 4px solid #4CAF50;
    }
    .stTextInput input {
        border-radius: 20px;
        padding: 12px 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 HealthAssist AI")
        st.markdown("---")
        st.markdown("Describe your symptoms for AI-powered guidance")
        st.markdown("---")
        st.markdown("⚠️ **Disclaimer**: Consult doctors for medical advice")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_input = ""
            st.session_state.processing = False
            st.rerun()
    
    # Main chat interface
    st.title("💬 Health Chatbot")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        if chat['type'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Assistant:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # User input with form to prevent multiple submissions
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your health concern...",
            placeholder="Example: headache, fever, stomach pain...",
            key="user_input"
        )
        submit_button = st.form_submit_button("Send")
    
    # Process input only when form is submitted and input is not empty
    if submit_button and user_input.strip():
        # Prevent processing the same input multiple times
        if user_input.strip() != st.session_state.last_input:
            st.session_state.processing = True
            st.session_state.last_input = user_input.strip()
            
            # Add user message to history
            st.session_state.chat_history.append({
                'type': 'user',
                'message': user_input.strip()
            })
            
            # Get bot response (only once)
            if 'bot' in st.session_state and st.session_state.bot_initialized:
                with st.spinner("Analyzing your symptoms..."):
                    bot_response = st.session_state.bot.get_response(user_input.strip())
            else:
                bot_response = "Chatbot is not available. Please check the dataset."
            
            # Add bot response to history
            st.session_state.chat_history.append({
                'type': 'bot', 
                'message': bot_response
            })
            
            st.session_state.processing = False
            
            # Rerun to update chat display
            st.rerun()

if __name__ == "__main__":
    main()