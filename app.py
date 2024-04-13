from flask import Flask, request, render_template
from src.pipeline import CustomData, PredictionPipeline



application=Flask(__name__)

app=application


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        data=CustomData(
            age=float(request.form.get('age')),
            balance=float(request.form.get('balance')),
            hascrcard=float(request.form.get('hascrcard')),
            isactivemember=float(request.form.get('isactivemember')),
            estimatedsalary=float(request.form.get('estimatedsalary')),
            geography=request.form.get('geography'),
            gender=request.form.get('gender'),
            tenure=int(request.form.get('tenure')),
            numofproducts=int(request.form.get('numofproducts'))
        )
        #transfrom data into DataFrame get ready for prediction
        predict_df= data.get_data_as_df()
        print(predict_df)
        print("Before Prediction")
        
        prediction_pipeline= PredictionPipeline()
        print("Mid Prediction")
        
        result=prediction_pipeline.predict(predict_df)
        print("after Prediction")
        
        return render_template('home.html', result=result[0])
    else: 
        return render_template('home.html')
    
    
if __name__ == "__main__":
    app.run()
