# MultiTool Chatbot

## Overview

This repository contains the code for a MultiTool Chatbot built using OpenAI's GPT-4 language model and Streamlit for the user interface. The chatbot is designed to assist users by providing answers in the Serbian language and leveraging various tools for tasks such as web scraping, document uploading, and semantic search.

## Features

- **Web Scraping:** Users can input a URL, and the chatbot will scrape the content of the website, convert it to a PDF, and upload it to OpenAI for further processing.

- **Document Upload:** Users can upload a file, and the chatbot will upload it to OpenAI for embedding and analysis.

- **Semantic Search:** The chatbot performs a hybrid search, combining semantic search with BM25 encoding, to provide relevant information based on user queries.

- **Chat Management:** Users can create, select, and manage different chats, each with its own namespace.

## Prerequisites

Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install openai streamlit beautifulsoup4 pdfkit pinecone
```

Additionally, you need to set environment variables for the OpenAI and Pinecone API keys.

## Usage

1. Run the Streamlit app:

```bash
streamlit run your_script_name.py
```

2. Open the provided URL in your browser to access the MultiTool Chatbot.

3. Use the chat interface to interact with the chatbot, ask questions, and utilize the various tools.

## Configuration

- **OpenAI API Key:** Set your OpenAI API key as an environment variable.
  
- **Pinecone API Key:** Set your Pinecone API key as an environment variable.

- **Other Configurations:** Update any other configuration details in the script, such as the path to the `wkhtmltopdf` executable.

## Notes

- The code includes error handling for better user experience.

- Ensure that your OpenAI account has access to the required models and resources.

- Use the chatbot responsibly and follow ethical guidelines for AI usage.

Feel free to explore and customize the code to suit your specific needs!