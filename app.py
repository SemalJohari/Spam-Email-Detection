from flask import Flask, render_template, request
import pickle as pk
import re
import string

app = Flask(__name__)

model = pk.load(open('model.pkl', 'rb')) 
vectorizer = pk.load(open('vectorizer.pkl', 'rb')) 

introduction = """
    <h3>About This Project:</h3>
    <p>This is a Spam Email Detection application made using a Multinomial Naive Bayes model.</p>
    <p>The model has been trained on the Kaggle dataset of spam and ham emails. It classifies</p> 
    <p>emails as either <strong>Spam</strong> or <strong>Ham</strong> based on their content.</p>
    <p>You can enter your email and get it checked for spam.</p>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == "POST":
        email_content = request.form.get("email")
        
        if email_content:
            email_processed = preprocess_email(email_content)
            email_vectorized = vectorizer.transform([email_processed])
            prediction = model.predict(email_vectorized)

            prediction = "Spam" if prediction[0] == "spam" else "Not Spam"
        else:
            prediction = "Please enter an email."

    return render_template("index.html", intro=introduction, prediction=prediction)

def preprocess_email(email_content):
    email_content = email_content.lower()
    email_content = email_content.strip()
    email_content = re.sub(r'http\S+|www\S+', '', email_content)
    email_content = email_content.translate(str.maketrans('', '', string.punctuation))
    email_content = re.sub(r'[^a-zA-Z0-9\s]', '', email_content)
    email_content = re.sub(r'\d+', '', email_content)
    
    return email_content

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

