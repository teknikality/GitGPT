# GitGPT

> A GPT-powered chatbot for intelligent interaction with GitLab documentation

## 📖 Overview

GitGPT is an intelligent chatbot that leverages GPT technology to provide accurate answers to queries about the GitLab Handbook and Direction pages. Built using a Retrieval-Augmented Generation (RAG) approach, it ensures responses are both precise and current.

## ✨ Features

- 🤖 Advanced GPT-based chatbot interface
- 🌍 Multi-language support (English and French) via DeepL API
- 📚 Real-time data retrieval from GitLab documentation
- 🔍 Efficient vector search using Pinecone embeddings
- 🎯 Focused on GitLab Handbook and Direction pages

## 🔧 Requirements

- Python 3.10+
- Virtual environment (recommended)
- API Keys:
  - OpenAI API key
  - Pinecone API key
  - DeepL API key (optional for translation features)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/teknikality/GitGPT.git
cd GitGPT
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# For macOS/Linux:
source venv/bin/activate
# For Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
DEEPL_API_KEY=your-deepl-api-key  # Optional for translation features
```

### 5. Launch the Application

```bash
streamlit run app/app.py
```

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.


---

Developed with ❤️ by Bharath