import os
import re
from tkinter.tix import Form
from flask import Flask,render_template,redirect,url_for,session,request,flash
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import pandas as pd

UPLOAD_FOLDER ='/project csc649'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'hello'

data = pd.read_csv('Medicalpremium.csv')
print(data)

from sklearn.model_selection import train_test_split
X=data[[ 'Age','Diabetes','BloodPressureProblems','AnyTransplants','AnyChronicDiseases','Height','Weight','KnownAllergies','HistoryOfCancerInFamily','NumberOfMajorSurgeries']] # Features
y=data['PremiumPrice']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)  #train model on training set
y_pred=clf.predict(X_test) #predict model on test set

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc = {'figure.figsize':(15,5)})
sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to graph\
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
plt.savefig('./static/output.png')
plt.show()


## flask part ##
@app.route('/', methods=['GET', 'POST'])
def input_pred():
    if request.method == 'POST' :
        age = request.form['age']
        diabetes = request.form['diabetes']
        bloodPressure = request.form['bloodpressure']
        transplant = request.form['transplant']
        chronicDisease = request.form['chronic']
        height = request.form['height']
        weight = request.form['weight']
        allergies = request.form['allergies']
        historyOfcancer = request.form['historyOfcancer']
        numOfSurgeries = request.form['surgeries']

        import csv
        from csv import writer

        fields = ['Age','Diabetes','BloodPressureProblems','AnyTransplants','AnyChronicDiseases','Height','Weight','KnownAllergies','HistoryOfCancerInFamily','NumberOfMajorSurgeries']
        #rows = [45,0,0,0,0,155,57,0,0,0]
        rows = [age,diabetes,bloodPressure,transplant,chronicDisease,height,weight,allergies,historyOfcancer,numOfSurgeries]
        filename = "PCD.csv"
            
        # writing to csv file 
        with open(filename, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
                
            # writing the fields 
            csvwriter.writerow(fields) 
                
            # writing the data rows 
        with open('PCD.csv', 'a') as csv_file:
            writer_object = writer(csv_file)
            writer_object.writerow(rows)
            csv_file.close()

        data = pd.read_csv('PCD.csv')

        input_prediction = clf.predict(data)
        print("Prediction: ",input_prediction)

        input_prediction2 = str(input_prediction)

        newstr = input_prediction2.replace("[", "")

        final_input = newstr.replace("]","")

        final_input = int(final_input)

        file = 'PCD.csv'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
            print("file deleted")
        else:
            print("file not found")

        return redirect(url_for('predictionResult',variable=final_input,data1=age,data2=diabetes,data3=bloodPressure,data4=transplant,data5=chronicDisease,data6=height,data7=weight,data8=allergies,data9=historyOfcancer,data10=numOfSurgeries))
    return render_template('medicalForm.html')

@app.route('/<string:variable>/<string:data1>/<string:data2>/<string:data3>/<string:data4>/<string:data5>/<string:data6>/<string:data7>/<string:data8>/<string:data9>/<string:data10>', methods=['GET', 'POST'])
def predictionResult(variable,data1,data2,data3,data4,data5,data6,data7,data8,data9,data10):
    return render_template('medicalForm.html',data=variable,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7,data8=data8,data9=data9,data10=data10)

if __name__ == '__main__':
      app.run(debug=True,port=5001)

file = './static/output.png'
if(os.path.exists(file) and os.path.isfile(file)):
    os.remove(file)
    print("file deleted")
else:
    print("file not found")