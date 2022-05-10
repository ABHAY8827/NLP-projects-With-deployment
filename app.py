import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]

    final_features = (features)
    from nltk.corpus import stopwords 
    from nltk.stem.porter import PorterStemmer
    import re
    ps= PorterStemmer()
    corpus = []
    for i in range(len(final_features)):
        review=re.sub('[^a-z-A-Z]', ' ',final_features[1])
        review=review.lower()
        review=review.split()
    
        review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
        review= ' '.join(review)
        corpus.append(review)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_v=TfidfVectorizer(max_features=1,ngram_range=(1,1))
    f_features= tfidf_v.fit_transform(corpus).toarray()
    f_features_re = f_features.reshape(1,2)
    f_features_re = f_features_re.reshape(1,-1)
    final_features_both = f_features_re,int(final_features[0])
    my_arr = np.array(final_features_both)
    mylist = my_arr.tolist()
    prediction = model.predict(mylist)
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)