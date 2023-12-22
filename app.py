# =[Modules dan Packages]========================
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from joblib import load

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================


# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
  return render_template('index.html')


# [Routing untuk API]
@app.route(
  "/api/prediksi", methods=['POST']
)  
def apiPrediksi():

  # POST data dari API
  if request.method == 'POST':
    # Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
    # $("#range_sepal_length").val();

    input_Lokasi = str(request.form.get('lokasi'))
    input_Luas_Tanah = str(request.form.get('luas_tanah'))
    input_Luas_Bangunan = str(request.form.get('luas_bangunan'))
    input_KT = str(request.form.get('kamar_tidur'))
    input_KM = str(request.form.get('kamar_mandi'))
    input_Listrik = str(request.form.get('listrik'))
    input_Garasi = str(request.form.get('garasi'))

    print(f'test : {input_KT}')

    # Nilai default untuk variabel input atau features (X) ke model
    if input_Lokasi == "Lokasi":
      input_Lokasi = "3"
    if input_KT == "KT":
      input_KT = "2"
    if input_KM == "KM":
      input_KM = "1"
    if input_Listrik == "listrik":
      input_Listrik = "2200"
    if input_Garasi == "Garasi":
      input_Garasi = "0"

    # Load data ke dalam DataFrame
    data = pd.read_csv('data_rumah_jabodetabek.csv', sep=';')
    # Preprocessing data untuk model rekomendasi
    # Encoder lokasi
    data['lokasi'] = data['lokasi'].map({
      'Kota Jakarta': 0,
      'Kota Bogor': 1,
      'Kabupaten Bogor': 2,
      'Kota Depok': 3,
      'Kota Tangerang': 4,
      'Kota Bekasi': 5,
      'Kabupaten Bekasi': 6
    })

    # Encoder garasi/carport
    data['garasi_carport'] = data['garasi_carport'].map({
      'Ada': 0,
      'Tidak ada': 1
    })

    # Membuat dataframe pandas
    df = pd.DataFrame(
      data={
        "lokasi": [input_Lokasi],
        "LT": [input_Luas_Tanah],
        "LB": [input_Luas_Bangunan],
        "KT": [input_KT],
        "KM": [input_KM],
        "listrik": [input_Listrik],
        "garasi_carport": [input_Garasi],
      })


    print(df)
    print(df[0:1])

    #Menampilkan Prediksi model adaboost
    hasil_prediksi = model.predict(df[0:1])[0]

    print(f'Hasil prediksi belum dikonversi:{hasil_prediksi}')

    hasil_prediksi_conv = np.exp(hasil_prediksi)

    print(f'Hasil prediksi setelah dikonversi:{hasil_prediksi_conv}')

    hasil_prediksi_conv = hasil_prediksi_conv / 1000000
    hasil_prediksi_conv = int(hasil_prediksi_conv)
    hasil_prediksi_conv = hasil_prediksi_conv * 1000000
    hasil_prediksi_conv = format(hasil_prediksi_conv, ',d')

    hasil_prediksi_conv = f"Rp{hasil_prediksi_conv}"

    # Return hasil prediksi dengan format JSON
    return jsonify({
      "prediksi": hasil_prediksi_conv,
    })


# =[Main]========================================

if __name__ == '__main__':
  # Load model yang telah ditraining
  model = load('dcp3.model')

  app.run(host="0.0.0.0", port=4000, debug=True)
  # app.run()
