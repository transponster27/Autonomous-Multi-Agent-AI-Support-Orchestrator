import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "edit_query" not in st.session_state:
    st.session_state.edit_query = ""
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

st.title("Document Assistant")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Settings")   
 
    # File Upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "png", "jpg"], label_visibility="collapsed")

    # Domain Selection
    st.subheader("Domain Context")
    upload_domain = st.selectbox(
        "Tag with domain:",
        ["None", "Business", "Medical", "Academic", "Banking", "Retail", "Personal", "Government", "Technical"],
        index=0,
        key="upload_domain"
    )
    
    if uploaded_file and st.button("Upload", use_container_width=True):
        with st.spinner("Uploading and indexing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            
            # Pass domain parameter
            data = {}
            if upload_domain != "None":
                data["domain"] = upload_domain.lower()
            
            response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
            result = response.json()
            
            if result.get("status") in ["uploaded", "success"]:
                st.success(result.get('message', 'File uploaded and indexed successfully'))
                st.info(f"Indexed {result.get('chunks_indexed', 0)} chunks")
            elif result.get("status") == "duplicate":
                st.warning(result.get('message', 'File already indexed'))
            else:
                st.error(f"Upload failed: {result.get('detail', 'Unknown error')}")
    
    st.divider()
    
    # Search Options - ONLY the toggle goes here
    st.subheader("Search Options")
    st.session_state.web_search_enabled = st.toggle(
        "Enable Web Search",
        value=st.session_state.web_search_enabled,
        help="Search the web in addition to uploaded documents"
    )
    
    if st.session_state.web_search_enabled:
        st.info("Web search is enabled. Queries will use the Research Agent.")

    st.divider()

    # Document Management
    st.subheader("Indexed Documents")
    
    try:
        docs_resp = requests.get(f"{BASE_URL}/documents")
        if docs_resp.status_code == 200:
            docs_data = docs_resp.json()
            documents = docs_data.get("documents", [])
            
            if not documents:
                st.caption("No documents indexed")
            else:
                for doc in documents:
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.caption(f"**{doc['filename']}**")
                        st.caption(f"_{doc['chunks']} chunks_")
                    with col2:
                        if st.button("Delete", key=f"del_{doc['filename']}"):
                            with st.spinner("Deleting..."):
                                del_resp = requests.delete(f"{BASE_URL}/documents/{doc['filename']}")
                                if del_resp.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed")
                
                # Delete All Button
                if st.button("Delete All Documents", use_container_width=True, type="secondary"):
                    with st.spinner("Deleting all..."):
                        del_all_resp = requests.delete(f"{BASE_URL}/documents")
                        if del_all_resp.status_code == 200:
                            st.success("All documents deleted!")
                            st.rerun()
                        else:
                            st.error("Failed")
    except Exception as e:
        st.caption("Cannot connect to server")
    
    st.divider()
    
    # Clear Chat Memory
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show citations for assistant messages
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    if citation.get("url"):
                        st.write(f"- [{citation['document']}]({citation['url']})")
                    else:
                        st.write(f"- {citation['document']} (Page {citation['page']})")
        
        # Edit button for user messages
        if message["role"] == "user":
            if st.button("Edit & Resend", key=f"edit_{i}"):
                st.session_state.edit_mode = True
                st.session_state.edit_query = message["content"]
                # Remove this message and all after it
                st.session_state.messages = st.session_state.messages[:i]
                st.rerun()

# ==========================================
# INPUT AREA
# ==========================================

if st.session_state.edit_mode:
    # Edit mode - show text input with pre-filled value
    st.info("Editing previous query. Modify and press Send.")
    
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        new_query = st.text_input(
            "Query:",
            value=st.session_state.edit_query,
            key="edit_input",
            label_visibility="collapsed"
        )
    with col2:
        send_clicked = st.button("Send", use_container_width=True, type="primary")
    
    if send_clicked and new_query:
        st.session_state.edit_mode = False
        st.session_state.edit_query = ""
        prompt = new_query
        process_query = True
    else:
        process_query = False
else:
    # Normal mode - show chat input
    prompt = st.chat_input("Ask a question...")
    process_query = bool(prompt)

# ==========================================
# PROCESS QUERY
# ==========================================

if process_query and prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build params correctly here
                params = {
                    "q": prompt,
                    "session_id": "streamlit_user",
                    "web_search": st.session_state.web_search_enabled
                }
                
                if st.session_state.selected_domain:
                    params["domain"] = st.session_state.selected_domain
                
                response = requests.get(f"{BASE_URL}/query", params=params)
                
                if response.status_code != 200:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error: {error_detail}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {error_detail}"
                    })
                else:
                    result = response.json()
                    
                    answer = result.get("answer", "I could not generate a response.")
                    citations = result.get("citations", [])
                    agent_type = result.get("agent_type", "unknown")
                    web_search_used = result.get("web_search_used", False)
                    
                    # ✅ Show appropriate caption based on agent type and web search
                    if web_search_used:
                        st.caption("Research response (documents + web)")
                    elif agent_type == "research":
                        st.caption("Research response (documents only)")
                    elif agent_type == "conversational":
                        st.caption("Conversational response")
                    else:
                        domain_tag = f" | Domain: {st.session_state.selected_domain.capitalize()}" if st.session_state.selected_domain else ""
                        st.caption(f"Contextual response from documents{domain_tag}")
                    
                    st.markdown(answer)
                    
                    if citations:
                        with st.expander("Sources"):
                            for citation in citations:
                                if citation.get("url"):
                                    st.write(f"- [{citation['document']}]({citation['url']})")
                                else:
                                    st.write(f"- {citation['document']} (Page {citation['page']})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to server. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"Error: {str(e)}")