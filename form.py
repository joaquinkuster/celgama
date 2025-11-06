from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo, scaler y mapeo
scaler = joblib.load('scaler.pkl')
modelo = joblib.load('modelo_kmeans.pkl')
mapeo_gama = joblib.load('mapeo_gama.pkl')

@app.route('/', methods=['GET', 'POST'])
def formulario():
    if request.method == 'POST':
        ram = float(request.form['ram'])
        battery = float(request.form['battery'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        int_memory = float(request.form['int_memory'])

        entrada_df = pd.DataFrame([[ram, battery, px_height, px_width, int_memory]],
            columns=['ram', 'battery_power', 'px_height', 'px_width', 'int_memory'])

        entrada_scaled = scaler.transform(entrada_df)
        cluster = modelo.predict(entrada_scaled)[0]
        gama = mapeo_gama[cluster]

        return f'<h2>Tu celular sería de: {gama}</h2>'

    return '''
        <form method="post">
            RAM (MB): <input name="ram"><br>
            Batería (mAh): <input name="battery"><br>
            Altura de pantalla (px): <input name="px_height"><br>
            Ancho de pantalla (px): <input name="px_width"><br>
            Memoria interna (GB): <input name="int_memory"><br>
            <input type="submit" value="Calcular gama">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)