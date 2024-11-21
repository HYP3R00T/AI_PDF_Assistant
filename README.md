# 🤖 Personal PDF Assistant: AI-Powered Document Analysis 📄

## 🌟 Project Overview

Personal PDF Assistant is an innovative AI-powered application that transforms how you interact with PDF documents. Leveraging cutting-edge machine learning technologies, this tool allows you to upload a PDF and engage in an intelligent conversation about its contents.

![Project Demo GIF](https://placeholdit.top/demo.gif)

## ✨ Key Features

- 📚 **Intelligent PDF Analysis**
  - Upload any PDF document
  - Ask complex questions about the document's content
  - Receive precise, context-aware responses

- 🧠 **Advanced AI Technology**
  - Powered by HuggingFace's Falcon-7B language model
  - Utilizes state-of-the-art embedding techniques
  - Supports both CPU and GPU computation

- 💻 **Easy-to-Use Interface**
  - Gradio-based web application
  - Intuitive document upload and chat interface
  - Responsive design

## 🚀 Technology Stack

- **Language**: Python 3.9+
- **AI Framework**:
  - HuggingFace Transformers
  - LangChain
- **Web Interface**: Gradio
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: Chroma

## 🔧 Prerequisites

- Python 3.9+
- Poetry
- HuggingFace API Token
- (Optional) CUDA-compatible GPU

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/personal-pdf-assistant.git
cd personal-pdf-assistant
```

### 2. Install Dependencies

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 3. Configure Environment

Create a `.env` file in the project root:

```text
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
```

## 🏃 Running the Application

```bash
poetry run python app.py
```

## 📖 Usage Guide

1. Open the Gradio web interface
2. Navigate to the "Document Upload" tab
3. Upload your PDF
4. Switch to the "Chat" tab
5. Ask questions about your document!

### Example Queries

- "Summarize the main points of this document"
- "What are the key takeaways from section 2?"
- "Extract the most important statistics"

## 🤝 How It Works

1. **Document Processing**
   - PDF is loaded and split into manageable chunks
   - Text chunks are converted to vector embeddings
   - Embeddings are stored in a vector database

2. **Intelligent Retrieval**
   - When you ask a question, the system:
     - Converts your query to an embedding
     - Retrieves most relevant document chunks
     - Generates a contextual response

## 🔬 Performance Optimization

- **GPU Acceleration**: Automatic GPU detection
- **Efficient Embedding**: Sentence Transformer models
- **Smart Retrieval**: Maximum Marginal Relevance (MMR) search

## 🌈 Customization

Easily modify `worker.py` to:

- Change language model
- Adjust embedding strategies
- Tune retrieval parameters

## 🤖 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - [Your Email/LinkedIn]

Project Link: [https://github.com/yourusername/personal-pdf-assistant](https://github.com/yourusername/personal-pdf-assistant)

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Gradio](https://www.gradio.app/)

---

⭐ Don't forget to star the repository if you find it helpful! 🌟
