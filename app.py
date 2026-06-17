import streamlit as st
import requests

st.title("🩺 Retinopathy Detection System")

uploaded_file = st.file_uploader("Upload retina image")

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("Analyze"):

        with st.spinner("Processing..."):

            files = {"file": uploaded_file.getvalue()}
            res = requests.post("http://127.0.0.1:8000/analyze", files=files)

            data = res.json()

            st.success("Analysis Complete")

            st.subheader("Results")
            st.write(f"**Stage:** {data['stage']}")
            st.write(f"**Condition:** {data['condition']}")
            st.write(f"**Confidence:** {data['confidence']}")

            st.subheader("Recommendation")
            st.write(data["recommendation"])
