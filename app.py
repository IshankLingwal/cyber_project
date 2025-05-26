import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# ----------------------------
# Load and preprocess data
# ----------------------------
data = pd.read_csv("spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['not spam', 'spam'])

mess = data['Message']
cat = data['Category']

# ----------------------------
# Split data
# ----------------------------
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2
)

# ----------------------------
# Feature extraction
# ----------------------------
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# ----------------------------
# Train model
# ----------------------------
model = MultinomialNB()
model.fit(features, cat_train)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("📨 Spam Detector")
st.write("Enter a message below to check if it's spam or not spam.")

# ----------------------------------------
# Single message prediction with confidence
# ----------------------------------------
st.subheader("Single Message with Confidence Score")
user_message = st.text_input("Enter your message:")

if st.button("Predict with Confidence"):
    if user_message.strip():
        input_features = cv.transform([user_message]).toarray()
        proba = model.predict_proba(input_features)[0]
        labels = model.classes_

        st.write(f"Prediction: **{labels[proba.argmax()]}**")

        fig, ax = plt.subplots()
        ax.bar(labels, proba, color=['green', 'red'])
        ax.set_ylabel("Probability")
        ax.set_title("Spam Prediction Confidence")
        st.pyplot(fig)
    else:
        st.error("Please enter a valid message.")

# ----------------------------
# Batch message input
# ----------------------------
with st.expander("📋 Textarea for Multiple Messages"):
    multi_input = st.text_area("Enter multiple messages (one per line):")
    
    if st.button("Predict Batch"):
        messages = [line.strip() for line in multi_input.strip().split("\n") if line.strip()]
        
        if messages:
            inputs = cv.transform(messages).toarray()
            predictions = model.predict(inputs)
            
            for msg, pred in zip(messages, predictions):
                st.write(f"**Message**: {msg}")
                st.write(f"Prediction: :blue[{pred}]\n---")
        else:
            st.error("Please enter valid messages.")

# ----------------------------
# --- Upload a TXT file for single document prediction ---
with st.expander("📄 Upload .txt File for Prediction"):
    uploaded_txt = st.file_uploader("Upload a .txt file containing a message", type=["txt"])
    
    if uploaded_txt is not None:
        # Read and decode text
        text_content = uploaded_txt.read().decode('utf-8').strip()
        
        if text_content:
            # Transform and predict
            input_features = cv.transform([text_content]).toarray()
            proba = model.predict_proba(input_features)[0]
            labels = model.classes_

            # Show prediction
            st.write(f"Prediction: **{labels[proba.argmax()]}**")

            # Plot confidence
            fig, ax = plt.subplots()
            ax.bar(labels, proba, color=['green', 'red'])
            ax.set_ylabel("Probability")
            ax.set_title("Spam Prediction Confidence")
            st.pyplot(fig)
        else:
            st.warning("The uploaded file is empty.")


# ----------------------------
# LIME Explanation Function
# ----------------------------
def show_lime_explanation(model, vectorizer, class_names, text_instance):
    """
    Optimized LIME explanation for Streamlit
    Args:
        model: Trained classifier with predict_proba
        vectorizer: Fitted text vectorizer (e.g., TfidfVectorizer)
        class_names: List of class labels (e.g., ['not spam', 'spam'])
        text_instance: Input string (user message)
    """
    # Create explainer
    explainer = LimeTextExplainer(class_names=class_names)

    # Prediction function for LIME
    predict_fn = lambda x: model.predict_proba(vectorizer.transform(x))

    with st.spinner("Generating explanation..."):
        explanation = explainer.explain_instance(
            text_instance,
            predict_fn,
            num_features=10,  # limit to avoid overload
            top_labels=1
        )

    # Display matplotlib explanation
    try:
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
    except Exception as e:
        st.error("Error displaying matplotlib figure.")
        st.text(str(e))

    # Display HTML explanation
    try:
        html = explanation.as_html()
        st.components.v1.html(html, height=600, scrolling=True)
    except Exception as e:
        st.warning("Could not render HTML explanation.")
        st.text(str(e))
