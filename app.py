import streamlit as st
import joblib
import pandas as pd
from PyPDF2 import PdfReader

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="DocuSense AI",
    page_icon="📄",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- HEADER ---------------- #

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image("logo.png", width=420)

st.markdown(
    """
    <p style='text-align:center;color:gray;font-size:20px;margin-top:-15px;'>
    AI-Powered Document Classification System
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT MODE ---------------- #

input_mode = st.radio(
    "Choose Input Method",
    ["Type Text", "Upload Document"]
)

user_text = ""

# ---------------- TYPE TEXT ---------------- #

if input_mode == "Type Text":

    user_text = st.text_area(
        "Enter Text",
        height=120,
        placeholder="Paste or type your text here..."
    )

# ---------------- UPLOAD DOCUMENT ---------------- #

else:

    uploaded_file = st.file_uploader(
        "Upload TXT or PDF",
        type=["txt", "pdf"]
    )

    if uploaded_file is not None:

        if uploaded_file.type == "text/plain":

            user_text = uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/pdf":

            pdf_reader = PdfReader(uploaded_file)

            pdf_text = ""

            for page in pdf_reader.pages:

                extracted = page.extract_text()

                if extracted:
                    pdf_text += extracted + "\n"

            user_text = pdf_text

        st.success("File uploaded successfully")

        word_count = len(user_text.split())
        char_count = len(user_text)

        col1, col2 = st.columns(2)

        col1.metric("Words", word_count)
        col2.metric("Characters", char_count)

# ---------------- COUNTS ---------------- #

if input_mode == "Type Text":

    word_count = len(user_text.split()) if user_text else 0
    char_count = len(user_text)

    col1, col2 = st.columns(2)

    col1.metric("Words", word_count)
    col2.metric("Characters", char_count)

# ---------------- BUTTONS ---------------- #

col1, col2 = st.columns(2)

predict_clicked = col1.button("Predict")

clear_clicked = col2.button("Clear")

if clear_clicked:
    st.rerun()

# ---------------- PREDICTION ---------------- #

if predict_clicked:

    if user_text.strip():

        text_vector = vectorizer.transform([user_text])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        confidence = max(probabilities) * 100

        categories = model.classes_

        prob_df = pd.DataFrame({
            "Category": categories,
            "Probability (%)": probabilities * 100
        })

        prob_df["Probability (%)"] = (
            prob_df["Probability (%)"]
            .round(2)
        )

        prob_df = prob_df.sort_values(
            by="Probability (%)",
            ascending=False
        )

        # -------- RESULT CARD -------- #

        st.markdown("---")

        st.subheader("Classification Result")

        st.success(
            f"Category : {prediction.title()}"
        )

        st.info(
            f"Model Confidence : {confidence:.2f}%"
        )

        if confidence >= 90:
            st.success("Status : High Confidence")

        elif confidence >= 70:
            st.info("Status : Medium Confidence")

        else:
            st.warning("Status : Low Confidence")

        if confidence < 60:

            st.warning(
                "The model is not very certain about this prediction."
            )

        # -------- CHART -------- #

        st.subheader("Category Probabilities")

        chart_df = prob_df.set_index("Category")

        st.bar_chart(chart_df)

        st.dataframe(
            prob_df,
            use_container_width=True
        )

    else:

        st.warning(
            "Please enter text or upload a document."
        )

# ---------------- SUGGESTIONS ---------------- #

st.markdown("---")

with st.expander("Sample Test Cases"):

    st.markdown("""
**1.** The company reported a strong increase in quarterly profits after expanding into new markets.

**2.** Manchester United secured a dramatic victory after scoring in the final minutes of the match.

**3.** Researchers unveiled a new artificial intelligence system capable of understanding complex language.

**4.** The actor received critical acclaim for their performance in the latest blockbuster film.

**5.** Parliament members debated the proposed economic reforms during a lengthy session.

**6.** A major software company launched a new cloud computing platform for businesses.

**7.** The national team advanced to the tournament finals after a dominant performance.

**8.** The government announced a new policy aimed at reducing unemployment across the country.

**9.** Scientists developed a new machine learning model capable of detecting diseases earlier.

**10.** The movie broke box office records during its opening weekend.
""")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption("""

TF-IDF Vectorization + Logistic Regression

BBC News Dataset (2225 Articles)
""")