import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchResults
import sqlite3

# Set Anthropic API Key 
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""


# Tool 1: Simple Legal Chatbot 
@tool("simple_chatbot")
def simple_chatbot(query: str) -> str:
    """Answer general legal questions using Claude model from Anthropic."""
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)  
    response = llm.invoke( "You are a legal expert with a deep understanding of law. "
        "You are to only respond to legal-related questions. "
        "If a question is not legal in nature, you should inform the user that you only handle legal queries. "
        "Answer this question in the context of law: "
        f"{query}")
    return response

# Tool 2: RAG Tool (Document-based Q&A)
def setup_rag_agent(documents):
    """Set up FAISS-based document retrieval."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(documents, embeddings)
    retriever = vector_store.as_retriever()

    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

@tool("rag_tool")
def rag_tool(query: str) -> str:
    """Answer questions based on uploaded documents."""
    prompt = (
        "You are a legal expert. You should only answer questions related to legal topics. "
        "If the query is not legal, inform the user that you are restricted to answering only legal-related queries. "
        "Please answer this legal question based on the uploaded documents: "
        f"{query}"
    )
    if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
        return "Please upload documents and initialize the RAG agent first."
    return st.session_state.rag_chain.run(prompt)

# Tool 3: Tavily Search Agent
@tool("tavily_search")
def tavily_search(query: str) -> str:
    """
    Perform Tavily search by:
    1. Generating a detailed response using the LLM.
    2. Extracting important keywords from the LLM-generated response.
    3. Using the extracted keywords to perform a Tavily search.
    """
    # Step 1: Generate a detailed response using the LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    detailed_response_prompt = (
        "You are a legal expert. You should only perform legal-related searches. "
        "If the query is not legal, inform the user that you are restricted to legal queries. "
        "Use the following query to extract legal-related terms and perform a Tavily search: "
        f"{query}"
    )
    detailed_response = llm.invoke(detailed_response_prompt).content.strip()

    # Step 2: Extract important keywords from the generated response
    keyword_extraction_prompt = (
        f"Given the detailed response below, extract only the most important legal terms or concepts relevant to the query.Focus on specific legal topics such as definitions, professional requirements, concepts, and processes that may help in a focused search."
        f"Provide the keywords as a comma-separated list:\n{detailed_response}"
    )
    extracted_keywords = llm.invoke(keyword_extraction_prompt).content.strip()

    # Step 3: Use Tavily Search with the extracted keywords
    tavily_tool = TavilySearchResults(max_results=3)
    search_results = tavily_tool.run(extracted_keywords)

    # Step 4: Format results for display
    formatted_results = ""
    if isinstance(search_results, list):  # Ensure results is a list
        for idx, result in enumerate(search_results, 1):
            url = result.get("url", "No URL available")
            content = result.get("content", "No content available")
            formatted_results += (
                f"**Result {idx}:**\n"
                f"- **Content:** {content}\n"
                f"- **URL:** [Click here]({url})\n\n"
            )
    else:
        formatted_results = "No valid results returned from Tavily Search."
    
    st.write("Results with Sources")
    st.markdown(formatted_results)

    # Step 5: Return the final response
    final_response = (
        f"### Detailed Response:\n\n{detailed_response}\n\n"
        f"### Extracted Keywords:\n\n{extracted_keywords}\n\n"
        f"### Search Results:\n\n{formatted_results}"
    )
    return final_response

# Tool 4:Database Agent
@tool("database_agent")
def database_agent(query: str) -> str:
    """
    Query the legal database for relevant information based on the user's question.
    """
    conn = sqlite3.connect("legal_assistant.db")
    cursor = conn.cursor()
    
    # Perform keyword-based search in the database
    cursor.execute("""
    SELECT title, content FROM legal_articles WHERE keywords LIKE ?
    """, ('%' + query + '%',))  # Simple keyword match
    
    results = cursor.fetchall()
    conn.close()
    
    # Format results
    if results:
        formatted_results = "\n".join([f"**{title}:** {content[:300]}..." for title, content in results])
        return f"Here are the relevant articles from the database:\n\n{formatted_results}"
    else:
        return "No relevant results found in the database."

# Tool 5: DuckDuckGo Search Agent
@tool("duckduckgo_search")
def duckduckgo_search(query: str) -> str:
    """
    Use DuckDuckGo to perform a search for legal-related queries and provide the results with links.
    """
    # Step 1: Generate a detailed response using the LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    detailed_response_prompt = (
      "You are a legal expert. Perform a search related only to legal topics. "
        "If the query is not legal, inform the user that you are restricted to legal-related queries. "
        "Perform the DuckDuckGo search based on this legal query: "
        f"{query}"
    )
    detailed_response = llm.invoke(detailed_response_prompt).content.strip()

    # Step 2: Extract important keywords from the generated response
    keyword_extraction_prompt = (
        f"Given the detailed response below, extract only the most important legal terms or concepts relevant to the query.Focus on specific legal topics such as definitions, professional requirements, concepts, and processes that may help in a focused search."
        f"Provide the keywords as a comma-separated list:\n{detailed_response}"
    )
    extracted_keywords = llm.invoke(keyword_extraction_prompt).content.strip()

    search_tool = DuckDuckGoSearchResults()
    search_results = search_tool.invoke(extracted_keywords)

    print("search duckduckgo:", search_results)
    print("keywords",extracted_keywords)
    
    st.write("Results with Sources")
    st.markdown(search_results)

    return search_results
    
# Define tools
tools = [
    Tool(
        name="Simple Chatbot",
        func=simple_chatbot,
        description="Use this to answer general legal questions only."
    ),
    Tool(
        name="RAG Tool",
        func=rag_tool,
        description="Use this to answer questions based on uploaded documents."
    ),
    Tool(
        name="Tavily Search",
        func=tavily_search,
        description="Perform keyword-based searches and return sources."
    ),
    Tool(
    name="Database Agent",
    func=database_agent,
    description="Query the database for legal articles based on keywords. Use this for specific legal topics."
    ),
    Tool(
        name="DuckDuckGo Search",
        func=duckduckgo_search,
        description="Perform a DuckDuckGo search for legal-related queries and return sources."
    )
]

# Initialize the Zero-Shot-React Agent
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)  # Replace with your model
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit Interface
st.title("Law Assistant Chatbot")
st.sidebar.title("Agent Options")

# Sidebar for tool selection
tool_choice = st.sidebar.radio("Choose an Agent", ["Simple Chatbot", "RAG Agent (Upload Documents)", "Tavily Search", "Database Agent","DuckDuckGo Search"])

# Tool 1: Simple Chatbot
if tool_choice == "Simple Chatbot":
    st.header("Simple Legal Chatbot")
    user_query = st.text_input("Enter your legal question:")
    if user_query:
        response = agent.run(f"Use the Simple Chatbot to answer this: {user_query}")
        st.write("Answer:", response)

# Tool 2: RAG Agent
elif tool_choice == "RAG Agent (Upload Documents)":
    st.header("Document-based Legal Q&A")
    
    # Upload documents
    uploaded_files = st.file_uploader(
        "Upload legal documents (TXT files only)", accept_multiple_files=True, type=["txt"]
    )

    if uploaded_files:
        documents = [file.read().decode("utf-8") for file in uploaded_files]
        st.session_state.rag_chain = setup_rag_agent(documents)
        st.write("Documents uploaded successfully and RAG Agent initialized!")

    user_query = st.text_input("Ask a question based on uploaded documents:")
    if user_query:
        response = agent.run(f"Use the RAG Tool to answer this: {user_query}")
        st.write("Answer:", response)

# Tool 3: Tavily Search
elif tool_choice == "Tavily Search":
    st.header("Keyword-based Legal Search with Sources")
    user_query = st.text_input("Enter your search query:")
    if user_query:
        response = agent.run(f"Use the Tavily Search to answer this: {user_query}")
        st.markdown(response)  # Display results with links and markdown formatting

# Tool 4: Database Agent
if tool_choice == "Database Agent":
    st.header("Structured Legal Information from Database")
    user_query = st.text_input("Enter your legal question or topic:")
    if user_query:
        response = database_agent(user_query)  # Directly invoke the database_agent
        st.write("Answer:", response)

# Tool 5: DuckDuckGo Search
elif tool_choice == "DuckDuckGo Search":
    st.header("DuckDuckGo Legal Search")
    user_query = st.text_input("Enter your search query:")  # Define user_query here
    if user_query:
        response = agent.run(f"Use the DuckDuckGo Search to answer this: {user_query}")
        st.write("Answer:", response)