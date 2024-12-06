import streamlit as st

# Home Page
st.title("Welcome to RedditAI")
st.subheader("Let's analyze our favorite Subreddits & Posts!")

# Initialize session state for role selection
if "role" not in st.session_state:
    st.session_state["role"] = "user"  # Default role is 'user'

# Sidebar for role selection
st.sidebar.title("Role Selection")
previous_role = st.session_state["role"]  # Store the previous role
selected_role = st.sidebar.selectbox(
    "Choose your role:",
    options=["user", "special"],
    index=["user", "special"].index(st.session_state["role"]),  # Use the current role as default
)

# Update the role and trigger rerun if it changes
if selected_role != previous_role:
    st.session_state["role"] = selected_role
    st.rerun()

st.image("assets/images/reddit-title.webp", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)