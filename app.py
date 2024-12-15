from flask import Flask, render_template, request


app=Flask(__name__)


with open('./LLMModel.h5', 'wb') as file:
    chain=file.load()




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', method=["post"])
def submit():
    feature_names = ['0']
    features= [float(request.form(f) for f in feature_names)]
    input =[features]

    ## Make the predictions
    response=chain.predict(input)

    return render_template("index.html", data=response)




if __name__=="main":
    app.run()