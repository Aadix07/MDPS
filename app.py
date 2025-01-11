# from flask import Flask, render_template, request
# import numpy as np
# import pickle

# pickle.load(open('models/diabetes.pkl', 'rb'))
# cancer_model = pickle.load(open('models/cancer.pkl', 'rb'))
# heart_model = pickle.load(open('models/heart.pkl', 'rb'))
# liver_model = pickle.load(open('models/liver.pkl', 'rb'))
# kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route("/diabetes", methods=['GET','POST'])
# def diabetes():
#     return render_template('diabetes.html')

# @app.route("/cancer", methods=['GET','POST'])
# def cancer():
#     return render_template('cancer.html')

# @app.route("/heart", methods=['GET','POST'])
# def heart():
#     return render_template('heart.html')

# @app.route("/kidney", methods=['GET','POST'])
# def kidney():
#     return render_template('kidney.html')

# @app.route("/liver", methods=['GET','POST'])
# def liver():
#     return render_template('liver.html')

# @app.route("/predict", methods=['GET','POST'])
# def predict():
#     if request.method == 'POST':
#         if(len([float(x) for x in request.form.values()])==8):
#             preg = int(request.form['pregnancies'])
#             glucose = int(request.form['glucose'])
#             bp = int(request.form['bloodpressure'])
#             st = int(request.form['skinthickness'])
#             insulin = int(request.form['insulin'])
#             bmi = float(request.form['bmi'])
#             dpf = float(request.form['dpf'])
#             age = int(request.form['age'])
            
#             data = np.array([[preg,glucose, bp, st, insulin, bmi, dpf, age]])
#             my_prediction = diabetes.predict(data)
            
#             return render_template('predict.html', prediction=my_prediction)
#         elif(len([float(x) for x in request.form.values()])==10):
#             Age = int(request.form['Age'])
#             Total_Bilirubin = float(request.form['Total_Bilirubin'])
#             Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
#             Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
#             Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
#             Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
#             Total_Protiens = float(request.form['Total_Protiens'])
#             Albumin = float(request.form['Albumin'])
#             Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
#             Gender_Male = int(request.form['Gender_Male'])

#             data = np.array([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])
#             my_prediction = liver_model.predict(data)
#             return render_template('predict.html', prediction=my_prediction)

#         elif(len([float(x) for x in request.form.values()])==13):
#             age = int(request.form['age'])
#             sex = int(request.form['sex'])
#             cp = int(request.form['cp'])
#             trestbps = int(request.form['trestbps'])
#             chol = int(request.form['chol'])
#             fbs = int(request.form['fbs'])
#             restecg = int(request.form['restecg'])
#             thalach = int(request.form['thalach'])
#             exang = int(request.form['exang'])
#             oldpeak = float(request.form['oldpeak'])
#             slope = int(request.form['slope'])
#             ca = int(request.form['ca'])
#             thal = int(request.form['thal'])

#             data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
#             data1 = np.array(data).reshape(1,-1)
#             my_prediction = heart_model.predict(data1)
#             return render_template('predict.html', prediction=my_prediction)

#         elif(len([float(x) for x in request.form.values()])==18):
#             age = float(int(request.form['age']))
#             bp = float(request.form['bp'])
#             al = float(request.form['al'])
#             su = float(request.form['su'])
#             rbc = int(request.form['rbc'])
#             pc = int(request.form['pc'])
#             pcc = int(request.form['pcc'])
#             ba = int(request.form['ba'])
#             bgr = float(request.form['bgr'])
#             bu = float(request.form['bu'])
#             sc = float(request.form['sc'])
#             pot = float(request.form['pot'])
#             wc = int(request.form['wc'])
#             htn = int(request.form['htn'])
#             dm = int(request.form['dm'])
#             cad = int(request.form['cad'])
#             pe = int(request.form['pe'])
#             ane = int(request.form['ane'])

#             data = [age,bp,al,su,rbc,pc,pcc,ba,bgr,bu,sc,pot,wc,htn,dm,cad,pe,ane]
#             data1 = np.array(data).reshape(1,-1)
#             my_prediction = kidney_model.predict(data1)
#             return render_template('predict.html', prediction=my_prediction)

#         elif(len([float(x) for x in request.form.values()])==26):
#             radius_mean = float(request.form['radius_mean'])
#             texture_mean = float(request.form['texture_mean'])
#             perimeter_mean = float(request.form['perimeter_mean'])
#             area_mean = float(request.form['area_mean'])
#             smoothness_mean = float(request.form['smoothness_mean'])
#             compactness_mean = float(request.form['compactness_mean'])
#             concavity_mean = float(request.form['concavity_mean'])
#             concave_points_mean = float(request.form['concave points_mean'])
#             symmetry_mean = float(request.form['symmetry_mean'])
#             radius_se = float(request.form['radius_se'])
#             perimeter_se = float(request.form['perimeter_se'])
#             area_se = float(request.form['area_se'])
#             compactness_se = float(request.form['compactness_se'])
#             concavity_se = float(request.form['concavity_se'])
#             concave_points_se = float(request.form['concave points_se'])
#             fractal_dimension_se = float(request.form['fractal_dimension_se'])
#             radius_worst = float(request.form['radius_worst'])
#             texture_worst = float(request.form['texture_worst'])
#             perimeter_worst = float(request.form['perimeter_worst'])
#             area_worst = float(request.form['area_worst'])
#             smoothness_worst = float(request.form['smoothness_worst'])
#             compactness_worst = float(request.form['compactness_worst'])
#             concavity_worst = float(request.form['concavity_worst'])
#             concave_points_worst = float(request.form['concave points_worst'])
#             symmetry_worst = float(request.form['symmetry_worst'])
#             fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

#             data = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,radius_se,perimeter_se,area_se,compactness_se,concavity_se,concave_points_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
#             data1 = np.array(data).reshape(1,-1)
#             my_prediction  = cancer_model.predict(data1)
#             return render_template('predict.html', prediction=my_prediction)












# if __name__ == "__main__":
#     app.run(debug=True)


# # from flask import Flask, render_template, request
# # import numpy as np
# # import pickle

# # # Load the models correctly
# # diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
# # cancer_model = pickle.load(open('models/cancer.pkl', 'rb'))
# # heart_model = pickle.load(open('models/heart.pkl', 'rb'))
# # liver_model = pickle.load(open('models/liver.pkl', 'rb'))
# # kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))

# # app = Flask(__name__)

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route("/diabetes", methods=['GET', 'POST'])
# # def diabetes():
# #     return render_template('diabetes.html')

# # @app.route("/cancer", methods=['GET', 'POST'])
# # def cancer():
# #     return render_template('cancer.html')

# # @app.route("/heart", methods=['GET', 'POST'])
# # def heart():
# #     return render_template('heart.html')

# # @app.route("/kidney", methods=['GET', 'POST'])
# # def kidney():
# #     return render_template('kidney.html')

# # @app.route("/liver", methods=['GET', 'POST'])
# # def liver():
# #     return render_template('liver.html')

# # @app.route("/predict", methods=['GET', 'POST'])
# # def predict():
# #     if request.method == 'POST':
# #         try:
# #             # Diabetes Model (8 features)
# #             if len([float(x) for x in request.form.values()]) == 8:
# #                 preg = int(request.form['pregnancies'])
# #                 glucose = int(request.form['glucose'])
# #                 bp = int(request.form['bloodpressure'])
# #                 st = int(request.form['skinthickness'])
# #                 insulin = int(request.form['insulin'])
# #                 bmi = float(request.form['bmi'])
# #                 dpf = float(request.form['dpf'])
# #                 age = int(request.form['age'])

# #                 data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
# #                 my_prediction = diabetes_model.predict(data)
# #                 return render_template('predict.html', prediction=my_prediction)

# #             # Liver Model (10 features)
# #             elif len([float(x) for x in request.form.values()]) == 10:
# #                 Age = int(request.form['Age'])
# #                 Total_Bilirubin = float(request.form['Total_Bilirubin'])
# #                 Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
# #                 Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
# #                 Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
# #                 Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
# #                 Total_Protiens = float(request.form['Total_Protiens'])
# #                 Albumin = float(request.form['Albumin'])
# #                 Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
# #                 Gender_Male = int(request.form['Gender_Male'])

# #                 data = np.array([[Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, 
# #                                   Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, Gender_Male]])
# #                 my_prediction = liver_model.predict(data)
# #                 return render_template('predict.html', prediction=my_prediction)

# #             # Heart Model (13 features)
# #             elif len([float(x) for x in request.form.values()]) == 13:
# #                 age = int(request.form['age'])
# #                 sex = int(request.form['sex'])
# #                 cp = int(request.form['cp'])
# #                 trestbps = int(request.form['trestbps'])
# #                 chol = int(request.form['chol'])
# #                 fbs = int(request.form['fbs'])
# #                 restecg = int(request.form['restecg'])
# #                 thalach = int(request.form['thalach'])
# #                 exang = int(request.form['exang'])
# #                 oldpeak = float(request.form['oldpeak'])
# #                 slope = int(request.form['slope'])
# #                 ca = int(request.form['ca'])
# #                 thal = int(request.form['thal'])

# #                 data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
# #                 data1 = np.array(data).reshape(1, -1)
# #                 my_prediction = heart_model.predict(data1)
# #                 return render_template('predict.html', prediction=my_prediction)

# #             # Kidney Model (18 features)
# #             elif len([float(x) for x in request.form.values()]) == 18:
# #                 age = float(int(request.form['age']))
# #                 bp = float(request.form['bp'])
# #                 al = float(request.form['al'])
# #                 su = float(request.form['su'])
# #                 rbc = int(request.form['rbc'])
# #                 pc = int(request.form['pc'])
# #                 pcc = int(request.form['pcc'])
# #                 ba = int(request.form['ba'])
# #                 bgr = float(request.form['bgr'])
# #                 bu = float(request.form['bu'])
# #                 sc = float(request.form['sc'])
# #                 pot = float(request.form['pot'])
# #                 wc = int(request.form['wc'])
# #                 htn = int(request.form['htn'])
# #                 dm = int(request.form['dm'])
# #                 cad = int(request.form['cad'])
# #                 pe = int(request.form['pe'])
# #                 ane = int(request.form['ane'])

# #                 data = [age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc, pot, wc, htn, dm, cad, pe, ane]
# #                 data1 = np.array(data).reshape(1, -1)
# #                 my_prediction = kidney_model.predict(data1)
# #                 return render_template('predict.html', prediction=my_prediction)

# #             # Cancer Model (26 features)
# #             elif len([float(x) for x in request.form.values()]) == 26:
# #                 radius_mean = float(request.form['radius_mean'])
# #                 texture_mean = float(request.form['texture_mean'])
# #                 perimeter_mean = float(request.form['perimeter_mean'])
# #                 area_mean = float(request.form['area_mean'])
# #                 smoothness_mean = float(request.form['smoothness_mean'])
# #                 compactness_mean = float(request.form['compactness_mean'])
# #                 concavity_mean = float(request.form['concavity_mean'])
# #                 concave_points_mean = float(request.form['concave points_mean'])
# #                 symmetry_mean = float(request.form['symmetry_mean'])
# #                 radius_se = float(request.form['radius_se'])
# #                 perimeter_se = float(request.form['perimeter_se'])
# #                 area_se = float(request.form['area_se'])
# #                 compactness_se = float(request.form['compactness_se'])
# #                 concavity_se = float(request.form['concavity_se'])
# #                 concave_points_se = float(request.form['concave points_se'])
# #                 fractal_dimension_se = float(request.form['fractal_dimension_se'])
# #                 radius_worst = float(request.form['radius_worst'])
# #                 texture_worst = float(request.form['texture_worst'])
# #                 perimeter_worst = float(request.form['perimeter_worst'])
# #                 area_worst = float(request.form['area_worst'])
# #                 smoothness_worst = float(request.form['smoothness_worst'])
# #                 compactness_worst = float(request.form['compactness_worst'])
# #                 concavity_worst = float(request.form['concavity_worst'])
# #                 concave_points_worst = float(request.form['concave points_worst'])
# #                 symmetry_worst = float(request.form['symmetry_worst'])
# #                 fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

# #                 data = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, 
# #                         concavity_mean, concave_points_mean, symmetry_mean, radius_se, perimeter_se, area_se, 
# #                         compactness_se, concavity_se, concave_points_se, fractal_dimension_se, radius_worst, 
# #                         texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, 
# #                         concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]
# #                 data1 = np.array(data).reshape(1, -1)
# #                 my_prediction = cancer_model.predict(data1)
# #                 return render_template('predict.html', prediction=my_prediction)

# #             else:
# #                 return render_template('index.html', message="Invalid number of input features. Please check your input.")

# #         except Exception as e:
# #             return render_template('index.html', message=f"An error occurred: {e}")

# # if __name__ == "__main__":
# #     app.run(debug=True)



# NEWWWW CODEEEEE

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Model definitions (paths and feature names)
# model_data = {
#     'diabetes': {
#         'path': 'models/diabetes.pkl',
#         'features': ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
#     },
#     'cancer': {
#         'path': 'models/cancer.pkl',
#         'features': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
#                      'compactness_mean', 'concavity_mean', 'concave points_mean', 
#                      'radius_se', 'area_se', 'concavity_se',
#                      'concave points_se', 'radius_worst', 'texture_worst',
#                      'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
#                      'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
#     },
#     'heart': {
#         'path': 'models/heart.pkl',
#         'features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
#                      'slope', 'ca', 'thal']
#     },
#     'liver': {
#         'path': 'models/liver.pkl',
#         'features': ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
#                      'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
#                      'Albumin_and_Globulin_Ratio']
#     },
#     'kidney': {
#         'path': 'models/kidney.pkl',
#         'features': ['age', 'bp', 'al', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc', 'pot', 'htn',
#                      'dm', 'ane']
#     }
# }

# # Load models
# models = {}
# for name, data in model_data.items():
#     try:
#         models[name] = pickle.load(open(data['path'], 'rb'))
#     except FileNotFoundError:
#         print(f"Error: Model file not found for {name} at {data['path']}")
#     except Exception as e:
#         print(f"Error loading model {name}: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route("/diabetes", methods=['GET','POST'])
# def diabetes():
#     return render_template('diabetes.html')

# @app.route("/cancer", methods=['GET','POST'])
# def cancer():
#     return render_template('cancer.html')

# @app.route("/heart", methods=['GET','POST'])
# def heart():
#     return render_template('heart.html')

# @app.route("/kidney", methods=['GET','POST'])
# def kidney():
#     return render_template('kidney.html')

# @app.route("/liver", methods=['GET','POST'])
# def liver():
#     return render_template('liver.html')

# @app.route("/predict", methods=['POST'])
# def predict():
#     model_name = request.form.get('model_name')

#     if model_name not in models:
#         return jsonify({'error': 'Invalid model name'}), 400

#     model_info = model_data[model_name]
#     model = models[model_name]

#     try:
#         # Input validation and data preparation
#         input_data = []
#         for feature in model_info['features']:
#             value = request.form.get(feature)
#             if value is None:
#                 return jsonify({'error': f'Missing feature: {feature}'}), 400
#             try:
#                 input_data.append(float(value))
#             except ValueError:
#                 return jsonify({'error': f'Invalid input for {feature}. Expecting a number.'}), 400

#         # Make prediction
#         prediction = model.predict(np.array([input_data]))

#         # Return prediction and model name to predict.html
#         return render_template('predict.html', prediction=prediction[0], model_name=model_name)

#     except Exception as e:
#         return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

##NEW CODEEE WITH AI CHATBOT#######
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import google.generativeai as genai
import os

app = Flask(__name__)

# Model definitions (paths and feature names)
model_data = {
    'diabetes': {
        'path': 'models/diabetes.pkl',
        'features': ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'dpf', 'age']
    },
    'cancer': {
        'path': 'models/cancer.pkl',
        'features': ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                     'compactness_mean', 'concavity_mean', 'concave points_mean',
                     'radius_se', 'area_se', 'concavity_se',
                     'concave points_se', 'radius_worst', 'texture_worst',
                     'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                     'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    },
    'heart': {
        'path': 'models/heart.pkl',
        'features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                     'slope', 'ca', 'thal']
    },
    'liver': {
        'path': 'models/liver.pkl',
        'features': ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                     'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                     'Albumin_and_Globulin_Ratio']
    },
    'kidney': {
        'path': 'models/kidney.pkl',
        'features': ['age', 'bp', 'al', 'rbc', 'pc', 'pcc', 'bgr', 'bu', 'sc', 'pot', 'htn',
                     'dm', 'ane']
    }
}

# Load models
models = {}
for name, data in model_data.items():
    try:
        models[name] = pickle.load(open(data['path'], 'rb'))
    except FileNotFoundError:
        print(f"Error: Model file not found for {name} at {data['path']}")
        # Consider raising the exception or exiting if a model is essential
    except Exception as e:
        print(f"Error loading model {name}: {e}")
        # Consider raising the exception or exiting if a model is essential

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not set in environment variables.")
    # You might want to exit the program or handle this error differently
else:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])  # Initialize chat history here

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET','POST'])
def cancer():
    return render_template('cancer.html')

@app.route("/heart", methods=['GET','POST'])
def heart():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET','POST'])
def kidney():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET','POST'])
def liver():
    return render_template('liver.html')

@app.route("/predict", methods=['POST'])
def predict():
    model_name = request.form.get('model_name')

    if model_name not in models:
        return jsonify({'error': 'Invalid model name'}), 400

    model_info = model_data[model_name]
    model = models[model_name]

    try:
        input_data = [float(request.form.get(feature)) for feature in model_info['features']]
        prediction = model.predict(np.array([input_data]))
        return render_template('predict.html', prediction=prediction[0], model_name=model_name)
    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

# Chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.form.get('message')
    if not user_message:
        return jsonify({'response': "Please provide a message."})

    try:
        global chat  # Access the global chat variable
        response = chat.send_message(user_message)

        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'response': f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)