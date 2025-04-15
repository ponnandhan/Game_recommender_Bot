from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("games.csv").fillna("")
df["search_text"] = (
    df["Genre"] + " " +
    df["Game Mode"] + " " +
    df["Platform"] + " " +
    df["Story Quality"].astype(str) + " " +
    df["Graphics Quality"].astype(str) + " " +
    df["Soundtrack Quality"].astype(str) + " " +
    df["User Review Text"]
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["search_text"])

def recommend_games(query, top_n=3):
    user_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    recommendations = recommend_games(user_input)
    
    if recommendations.empty:
        reply = "❌ Sorry, I couldn't find any matching games."
    else:
        reply = ""
        for _, row in recommendations.iterrows():
            reply += f"<b>{row['Game Title']}</b> ({row['Genre']} on {row['Platform']})<br>"
            reply += f"🎮 Mode: {row['Game Mode']}, 🎨 Graphics: {row['Graphics Quality']}, 📖 Story: {row['Story Quality']}<br>"
            reply += f"🗣️ Review: {row['User Review Text'][:150]}...<br><br>"
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(debug=True)
