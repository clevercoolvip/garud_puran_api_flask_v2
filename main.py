from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from config import data_path
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# databse setup for flask
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

if not os.path.exists("instance/db.sqlite3"):
    print("Not exist")
    with app.app_context():
        db.create_all()
else:
    app.app_context().push()
def add_data(item):
    db.session.add(item)
    db.session.commit()
    return True


#predicting the class of the query
data = pd.read_csv(data_path).iloc[:28, :]
output_data = pd.read_csv("penalty.csv")
output_dic = {}
for i in range(len(output_data)):
  output_dic[output_data.iloc[i, 0].lower()] = output_data.iloc[i, 1]

class_names = data.iloc[:, 0].to_list()
class_texts_embeddings = torch.load("class_embeddings.pt")

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_result(query):
    query_embedding = model.encode(query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, class_texts_embeddings)
    if torch.max(cos_scores) < 0.325:
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
        return jsonify({"punishment": result, "description": output_dic[result.lower()]})
    else:
        return jsonify({"punishment":"Bach gya tu", "description": "Maybe!"})


@app.route("/forum", methods=['GET'])
def get_latest_entries():
    entries = Confessions.query.order_by(Confessions.created_on.desc()).limit(10).all()
    return jsonify([entry.to_json() for entry in entries])


@app.route('/vote', methods=['POST'])
def print_req():
    data = request.get_json()
    confession_id = data.get("id")
    vote_type = data.get("voteType", "").lower()

    with db.session() as session:
        confession_data = session.get(Confessions, confession_id)

        if not confession_data:
            confession_data = Confessions(id=confession_id)
            

        if vote_type == 'upvote':
            confession_data.upvotes = (confession_data.upvotes or 0) + 1
        elif vote_type == 'downvote':
            confession_data.downvotes = (confession_data.downvotes or 0) + 1
        else:
            return jsonify({"error": "Invalid voteType"}), 400

        session.add(confession_data)
        session.commit()

        return jsonify({
            "message": "Vote counted",
            "upvotes": confession_data.upvotes,
            "downvotes": confession_data.downvotes
        })





if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)