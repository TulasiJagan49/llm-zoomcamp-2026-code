"""
Use Streamlit to create a UI for the LLM Zoomcamp assistant.
"""

import streamlit as st

from assistant import create_assistant
from db_save import save_conversation, save_feedback


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="LLM Zoomcamp Course Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LLM Zoomcamp Course Assistant")
st.caption("Ask questions about the LLM Zoomcamp course.")


# -----------------------------
# Initialize assistant
# -----------------------------
assistant = create_assistant()


# -----------------------------
# Initialize session state
# -----------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "response_time" not in st.session_state:
    st.session_state.response_time = None

if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = None

if "completion_tokens" not in st.session_state:
    st.session_state.completion_tokens = None

if "cost" not in st.session_state:
    st.session_state.cost = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False


# -----------------------------
# Question input
# -----------------------------
user_input = st.text_input(
    "Enter your question",
    placeholder="e.g. What is RAG?",
)


# -----------------------------
# Ask button
# -----------------------------
if st.button("Ask", type="primary", use_container_width=True):

    # Reset feedback for the new question
    st.session_state.feedback_given = False
    st.session_state.conversation_id = None

    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            # Ask assistant
            answer = assistant.rag(user_input)

            # Get information about the LLM call
            record = assistant.last_call

            # Save conversation
            conversation_id = save_conversation(
                record,
                user_input,
                "llm-zoomcamp",
            )

        # Save everything in session state
        st.session_state.answer = answer
        st.session_state.conversation_id = conversation_id
        st.session_state.response_time = record.response_time
        st.session_state.prompt_tokens = record.prompt_tokens
        st.session_state.completion_tokens = record.completion_tokens
        st.session_state.cost = record.cost

        st.success("Completed!")


# -----------------------------
# Display answer
# -----------------------------
if st.session_state.answer is not None:

    st.markdown("### Answer")

    st.markdown(st.session_state.answer)

    # Metrics
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Response time",
            f"{st.session_state.response_time:.2f}s",
        )

    with col2:
        st.metric(
            "Prompt tokens",
            st.session_state.prompt_tokens,
        )

    with col3:
        st.metric(
            "Completion tokens",
            st.session_state.completion_tokens,
        )

    with col4:
        st.metric(
            "Cost",
            f"${st.session_state.cost:.4f}",
        )

    # -----------------------------
    # Feedback section
    # -----------------------------
    if (
        st.session_state.conversation_id is not None
        and not st.session_state.feedback_given
    ):

        st.divider()

        st.markdown("**Was this answer helpful?**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "👍 Helpful",
                key=f"feedback_up_{st.session_state.conversation_id}",
                use_container_width=True,
            ):
                save_feedback(
                    st.session_state.conversation_id,
                    source="user",
                    score=1,
                )

                st.session_state.feedback_given = True
                st.rerun()

        with col2:
            if st.button(
                "👎 Not helpful",
                key=f"feedback_down_{st.session_state.conversation_id}",
                use_container_width=True,
            ):
                save_feedback(
                    st.session_state.conversation_id,
                    source="user",
                    score=-1,
                )

                st.session_state.feedback_given = True
                st.rerun()

    elif st.session_state.feedback_given:

        st.caption("Thanks for your feedback! 🙏")
