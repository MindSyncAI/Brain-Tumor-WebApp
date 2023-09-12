from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)
model = load_model('Brain_tumor_flask\model.h5')  # Replace with the actual path to your trained model

# OneHotEncoder setup (you can reuse the code you provided)
# ...

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result="No file selected.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No file selected.")

        if file:
            file_path = "temp.jpg"  # Save the uploaded image temporarily
            file.save(file_path)

            # Preprocess the image and perform the prediction
            img = Image.open(file_path)
            img = img.resize((128, 128))
            img = np.array(img)
            if img.shape == (128, 128, 3):
                x = np.array([img])
                res = model.predict(x)
                classification = np.where(res == np.amax(res))[1][0]
                confidence = res[0][classification] * 100
                result = f"{confidence:.2f}% Confidence This Is A {'Tumor' if classification == 0 else 'No, Not a Tumor'}"

            os.remove(file_path)  # Remove the temporary file

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
