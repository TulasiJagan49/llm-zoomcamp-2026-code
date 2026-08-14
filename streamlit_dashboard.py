'''
Run this file using this command:
streamlit run streamlit_dashboard.py --server.port 8502
Just in case, if we have the assistant running on
8501 port
'''
import pandas as pd
import streamlit as st

from dataclasses import asdict

from db_query import get_conversations, get_stats


st.title("Course Assistant Dashboard")

stats = get_stats()

col_1, col_2, col_3, col_4 = st.columns(4)
col_1.metric(label="Total requests", value=stats.total)
col_2.metric(label="Avg Response Time", value=f"{stats.avg_response_time:.2f}s")
col_3.metric(label="Total Cost", value=f"{stats.total_cost:.4f}")
col_4.metric(label="Average Token", value=f"{stats.avg_tokens:.0f}")


# Check for cost and response time over time
records = get_conversations(limit=100)
df = pd.DataFrame([asdict(r) for r in records])

st.subheader("Cost over time")
st.line_chart(data=df, x="timestamp", y="cost")

st.subheader("Response time over time")
st.line_chart(data=df, x="timestamp", y="response_time")

st.subheader("Recent conversations")
records = get_conversations(limit=20)

for record in records:
    st.write(f"**{record.prompt[:80]}...**")
    st.write(f"{record.answer[:200]}...")
    st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
    st.divider()