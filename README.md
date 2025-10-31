
# Task-2
# Install: pip install scikit-learn nltk

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is your name?": "I am an FAQ chatbot.",
    "How can I reset my password?": "Go to settings > reset password.",
    "What is your refund policy?": "We offer 30-day money-back guarantee.",
    "How do I contact support?": "Email us at support@example.com."
}

# Preprocess
questions = list(faqs.keys())
answers = list(faqs.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    return answers[idx]

# Example
user_q = "Tell me how to reset password"
print("User:", user_q)
print("Bot:", chatbot_response(user_q))
