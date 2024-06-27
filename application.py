from flask import Flask,request,render_template

from src.pipeline.test_pipeline import PredictPipeline,UserInputData
from src.utils import get_column_categories

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        userinput=UserInputData(
            brand=request.form.get('brand'),
            model=request.form.get('model'),
            model_year=request.form.get('model_year'),
            mileage=str(request.form.get('mileage')),
            fuel_type=request.form.get('fuel_type'),
            transmission=request.form.get('transmission'),
            engine=request.form.get('engine'),
            ext_col=request.form.get('ext_col'),
            int_col=request.form.get('int_col'),
            accident=request.form.get('accident'),
            clean_title=request.form.get('clean_title')
        )

        dataframe=userinput.createDataFrame()
        print(dataframe)

        model=PredictPipeline()
        y_pred=model.predict(dataframe)

        return render_template('predict.html',predict=y_pred[0])
    
    else:
        column_cat=get_column_categories()
        print(column_cat)
        return render_template('predict.html',categories=column_cat)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)