import streamlit as st

# Home Page
st.title("Welcome to our App")
st.write("Blabla!")

# Initialize session state for role selection
if "role" not in st.session_state:
    st.session_state["role"] = "user"  # Default role is 'user'

# Sidebar for role selection
st.sidebar.title("Role Selection")
previous_role = st.session_state["role"]  # Store the previous role
selected_role = st.sidebar.selectbox(
    "Choose your role:",
    options=["user", "MLDS-Staff"],
    index=["user", "MLDS-Staff"].index(st.session_state["role"]),  # Use the current role as default
)

# Update the role and trigger rerun if it changes
if selected_role != previous_role:
    st.session_state["role"] = selected_role
    st.rerun()

# Display the current role on the page
st.write(f"Your current role is: {st.session_state['role']}")

