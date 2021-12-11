from flask import Flask, render_template, request as requests
import numpy as np
import joblib
import requests as req
import schedule

app = Flask(__name__)

def kawal_corona():
    api_url = "https://api.kawalcorona.com/indonesia/"
    result = req.get(api_url).json()
    return result

data_indonesia = kawal_corona()
r_indonesia = schedule.every(2).seconds.do(kawal_corona)

@app.route("/")
def index():
    return render_template("index.html", data_indonesia=data_indonesia, alert="")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    if requests.method == "POST":
        model = requests.form.get('model')
        kasus_aktif = requests.form.get('kasus-aktif')
        kasus_baru = requests.form.get('kasus-baru')

        if kasus_aktif == "" or kasus_baru == "":
            return render_template('index.html', data_indonesia=data_indonesia, alert="Lengkapi Input!")
        elif int(kasus_aktif) > 574135 or int(kasus_aktif) < 7526:
            return render_template('index.html', data_indonesia=data_indonesia, alert="Mesin tidak dapat memprediksi karena input kasus aktif terlalu tinggi atau terlalu rendah!")
        elif int(kasus_baru) < 176 or int(kasus_baru) > 56757:
            return render_template('index.html', data_indonesia=data_indonesia, alert="Mesin tidak dapat memprediksi karena input kasus baru terlalu tinggi atau terlalu rendah!")

        try:
            prediction = preprocessDataAndPredict(model=model, kasus_aktif=kasus_aktif, kasus_baru=kasus_baru)
            return render_template('predict.html', data_indonesia=data_indonesia, prediction=prediction)

        except ValueError:
            # return "Masukkan Input yang valid!"
            print("asa yang dkjef")
            return render_template('index.html', data_indonesia=data_indonesia, alert="Lengkapi Input!")

    pass

def preprocessDataAndPredict(model, kasus_aktif, kasus_baru):
    # membuat test data
    test_data = [kasus_aktif, kasus_baru]
    test_data = np.array(test_data)
    test_data = test_data.reshape(1,-1)

    # membuka model
    if model == "Decision Tree Reggresion":
        filePath = "models/Covid-19-prediksi-sembuh-meninggal-baru-decision_tree_regresion.pkl"
    else:
        filePath = "models/Covid-19-prediksi-sembuh-meninggal-baru-random_forest_regresion.pkl"

    file = open(filePath, 'rb')

    # trainned model
    trained_model = joblib.load(file)

    # melakukan prediksi
    prediction = trained_model.predict(test_data)

    return prediction


if __name__ == "__main__":
    app.run(debug=True)