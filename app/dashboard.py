import streamlit as st

# Define authentication logic
def is_operator():
    return st.session_state.get("role") == "MLDS-Staff"

# Initialize session state for role selection
if "role" not in st.session_state:
    st.session_state["role"] = "user"  # Default role is 'user'

def is_user():
    return st.session_state.get("role") == "user"

pages = [st.Page("views/home.py", title="Home", icon="🏠", default=True),
         st.Page("views/dashboard_analysis.py", title="Reddit AI Dashboard", icon="🧠", default=False)]

if is_operator():
    pages.append(st.Page("views/special_page.py", title="Special Dashboard", icon="📈"))

page = st.navigation(pages)

page.run()
