from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from config import data_path
import os
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Confessions(db.Model):
    id = db.Column("ID", db.Integer, primary_key=True)
    created_on = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    user_query = db.Column(db.String(2500), nullable=False)
    predicted_class = db.Column(db.String(250), nullable=False)
    max_cosine_score = db.Column(db.Float, nullable=False)
    upvotes = db.Column(db.Integer, default=0)
    downvotes = db.Column(db.Integer, default=0)

    def to_json(self):
        return {
            'id':self.id,
            'user_query': self.user_query,
            'predicted_class': self.predicted_class,
            'max_cosine_score': self.max_cosine_score,
            'created_on': self.created_on.strftime('%Y-%m-%d %H:%M:%S'),
            'upvotes': self.upvotes,
            'downvotes': self.downvotes
        }
    
class Vote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confession_id = db.Column(db.Integer, db.ForeignKey('confessions.ID'))
    fingerprint = db.Column(db.String(128), nullable=False)
    vote_type = db.Column(db.String(10))
    __table_args__ = (
    db.UniqueConstraint('confession_id', 'fingerprint', name='unique_vote_per_user'),
    )

if not os.path.exists("instance/db.sqlite3"):
    print("Database not found, creating...")
    with app.app_context():
        db.create_all()
else:
    app.app_context().push()

def add_data(item):
    db.session.add(item)
    db.session.commit()
    return True

data = pd.read_csv(data_path).iloc[:28, :]
output_data = pd.read_csv("penalty.csv")
output_dic = {output_data.iloc[i, 0].lower(): output_data.iloc[i, 1] for i in range(len(output_data))}

class_names = data.iloc[:, 0].to_list()
model = SentenceTransformer('all-MiniLM-L6-v2')
# class_texts_embeddings = model.encode(data.iloc[:, 1].to_list(), convert_to_tensor=True)
# torch.save(class_texts_embeddings, "class_embeddings_v2.pt")
class_texts_embeddings = torch.load("class_embeddings_v2.pt")
print("Model saved")


def get_result(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, class_texts_embeddings)
    if torch.max(cos_scores) < 0.225:
        return "please enter some specific situation, it is too generic or it is not a sin", torch.max(cos_scores).item()
    else:
        predicted_class = torch.argmax(cos_scores).item()
        return class_names[predicted_class], torch.max(cos_scores).item()


@app.route('/', methods=['POST'])
def classify_text():
    query = request.get_json().get('text')
    result, max_cos_score = get_result(query)
    item = Confessions(user_query=query, predicted_class=result, max_cosine_score=max_cos_score)
    add_data(item)
    if " " not in result:
        return jsonify({"punishment": result, "description": output_dic.get(result.lower(), "No description available."), "score": max_cos_score})
    else:
        return jsonify({"punishment": "Bach gya tu", "description": "Maybe!", "score": max_cos_score})
    

@app.route("/forum", methods=['GET'])
def get_latest_entries():
    entries = Confessions.query.order_by(Confessions.created_on.desc()).limit(10).all()
    return jsonify([entry.to_json() for entry in entries])


@app.route('/vote', methods=['POST'])
def handle_vote():
    data = request.get_json()
    confession_id = data.get("id")
    vote_type = data.get("voteType", "").lower()
    fingerprint = data.get("fingerprint")
    if not fingerprint or not vote_type:
        return jsonify({"error": "Missing fingerprint or voteType"}), 400

    existing_vote = Vote.query.filter_by(confession_id=confession_id, fingerprint=fingerprint).first()

    if existing_vote:
        return jsonify({"message": "Already voted"}), 409

    confession = Confessions.query.get(confession_id)
    if not confession:
        return jsonify({"error": "Confession not found"}), 404

    if vote_type == 'upvote':
        confession.upvotes += 1
    elif vote_type == 'downvote':
        confession.downvotes += 1
    else:
        return jsonify({"error": "Invalid voteType"}), 400

    new_vote = Vote(confession_id=confession_id, fingerprint=fingerprint, vote_type=vote_type)
    add_data(new_vote)

    return jsonify({
        "message": "Vote recorded",
        "upvotes": confession.upvotes,
        "downvotes": confession.downvotes
    })

@app.route('/get-votes', methods=['POST'])
def get_user_votes():
    data = request.get_json()
    fingerprint = data.get('fingerprint')
    if not fingerprint:
        return jsonify({"error": "Missing fingerprint"}), 400

    votes = Vote.query.filter_by(fingerprint=fingerprint).all()
    result = {vote.confession_id: vote.vote_type for vote in votes}
    return jsonify({"voted": result})


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)