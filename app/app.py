from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# --- Load TFLite model ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "VGG19_model.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"--- ERROR: Could not load TFLite model ---")
    print(f"Error: {e}")
    interpreter = None

# Class names mapping
class_names = [
    'Corn Common Rust', 'Corn Gray Leaf', 'Corn Healthy', 'Corn Northern Leaf Blight',
    'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
    'Rice Brown Spot', 'Rice Healthy', 'Rice Leaf Blast', 'Rice Neck Blast',
    'Wheat Brown Rust', 'Wheat Healthy', 'Wheat Yellow Rust'
]

# --- Disease Information Database (in English) ---
disease_info = {
    'Corn Common Rust': {
        "description": "Common rust, caused by the fungus Puccinia sorghi, appears as cinnamon-brown, powdery pustules on both upper and lower leaf surfaces. It thrives in cool, moist conditions.",
        "treatment": "Use resistant corn hybrids. Apply foliar fungicides when rust is first detected, especially if conditions are favorable for disease development. Monitor fields regularly."
    },
    'Corn Gray Leaf': {
        "description": "Gray leaf spot is a fungal disease that causes long, narrow, rectangular lesions on the leaves, which are typically pale brown or gray. It's favored by warm, humid weather.",
        "treatment": "Plant resistant hybrids. Practice crop rotation and tillage to reduce residue where the fungus overwinters. Fungicide applications can be effective if applied early."
    },
    'Corn Healthy': {
        "description": "The plant appears to be healthy. No significant signs of disease are detected. Continue standard monitoring and care.",
        "treatment": "Maintain good agronomic practices, ensure proper nutrition and irrigation, and continue to monitor for any signs of stress or disease."
    },
    'Corn Northern Leaf Blight': {
        "description": "This disease produces long, elliptical, grayish-green or tan lesions on the leaves. It is one of the most significant foliar diseases of corn worldwide.",
        "treatment": "Planting resistant hybrids is the most effective management strategy. Fungicides may be necessary for susceptible hybrids in high-risk environments."
    },
    'Potato Early Blight': {
        "description": "Early blight, caused by Alternaria solani, creates dark, concentric lesions often described as 'target spots' on lower leaves. It can also affect stems and tubers.",
        "treatment": "Use certified disease-free seed. Apply fungicides preventively, especially during humid weather. Rotate crops and remove volunteer potato plants."
    },
    'Potato Healthy': {
        "description": "The plant appears to be healthy. No significant signs of disease are detected. Continue standard monitoring and care.",
        "treatment": "Maintain a consistent watering schedule, ensure soil is well-drained, and provide adequate nutrients. Monitor for pests and diseases regularly."
    },
    'Potato Late Blight': {
        "description": "Late blight is a devastating disease caused by the oomycete Phytophthora infestans. It causes large, dark, water-soaked lesions on leaves and stems.",
        "treatment": "Apply fungicides proactively, especially before periods of cool, wet weather. Use resistant varieties and ensure good field drainage. Destroy infected plants to prevent spread."
    },
    'Rice Brown Spot': {
        "description": "Brown spot is a fungal disease that causes numerous large spots on the leaves and glumes of the rice plant, leading to significant yield losses.",
        "treatment": "Ensure balanced nutrient management, particularly potassium. Use treated seeds and apply fungicides if the disease becomes severe."
    },
    'Rice Healthy': {
        "description": "The plant appears to be healthy. No significant signs of disease are detected. Continue standard monitoring and care.",
        "treatment": "Manage water levels effectively, provide balanced fertilization, and control weeds to maintain plant vigor and reduce disease risk."
    },
    'Rice Leaf Blast': {
        "description": "Rice blast, caused by the fungus Magnaporthe oryzae, creates diamond-shaped lesions with grayish centers and brown margins on leaves, which can kill the plant.",
        "treatment": "Plant resistant varieties. Manage nitrogen fertilizer application to avoid excess growth. Apply fungicides at the first sign of disease."
    },
    'Rice Neck Blast': {
        "description": "A severe form of rice blast that affects the 'neck' of the panicle, causing it to rot and break. This can result in complete loss of the grain head.",
        "treatment": "Similar to leaf blast, use resistant varieties and manage nitrogen. Protective fungicide applications during the heading stage are crucial for control."
    },
    'Wheat Brown Rust': {
        "description": "Also known as leaf rust, this disease produces small, circular to oval, orange-brown pustules on the upper surfaces of wheat leaves.",
        "treatment": "Planting resistant wheat varieties is the primary control method. Foliar fungicides can be effective if applied when the disease first appears."
    },
    'Wheat Healthy': {
        "description": "The plant appears to be healthy. No significant signs of disease are detected. Continue standard monitoring and care.",
        "treatment": "Follow recommended planting dates, use certified seed, and ensure proper fertilization and weed control to maintain a healthy crop."
    },
    'Wheat Yellow Rust': {
        "description": "Also called stripe rust, this disease is characterized by yellow-to-orange pustules that form distinct stripes on the leaves. It prefers cool, moist conditions.",
        "treatment": "Use of resistant varieties is the most economical and effective control method. Early application of fungicides is critical for managing susceptible varieties."
    }
}


# --- Yield and Price Data (Estimates) ---
yield_data = {
    # Average Yield (Tonnes per Hectare) and Average Price (INR per Tonne)
    'Corn': {'yield_per_hectare': 6, 'price_per_tonne': 21000},
    'Potato': {'yield_per_hectare': 25, 'price_per_tonne': 18000},
    'Rice': {'yield_per_hectare': 4, 'price_per_tonne': 26000},
    'Wheat': {'yield_per_hectare': 3.5, 'price_per_tonne': 23000},
}

# --- Utility: preprocess image ---
def preprocess_image(image):
    if interpreter is None:
        raise ValueError("Interpreter not loaded, cannot preprocess image.")
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    image = image.resize((width, height))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/developers')
def developers():
    return render_template('developers.html')
    
@app.route('/yield')
def yield_page():
    return render_template('yield.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return "Model not loaded. Please check server logs.", 500
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file selected.", 400

    try:
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(image)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        probabilities = output_data[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        
        predicted_class_name = class_names[predicted_class_index]
        prediction_confidence = f"{confidence:.2%}"

        info = disease_info.get(predicted_class_name, {
            "description": "Information not available.",
            "treatment": "Consult a local agricultural expert for guidance."
        })

        return render_template('result.html', 
                               prediction=predicted_class_name,
                               confidence=prediction_confidence,
                               description=info['description'],
                               treatment=info['treatment'])
    except Exception as e:
        return f"An error occurred during prediction: {e}", 500

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    try:
        crop_type = request.form['crop_type']
        area = float(request.form['area'])
        unit = request.form['unit']

        # Convert area to Hectares
        if unit == 'acre':
            area_in_hectares = area * 0.404686
        else: # hectare
            area_in_hectares = area

        # Calculate yield and price
        if crop_type in yield_data:
            data = yield_data[crop_type]
            calculated_yield = area_in_hectares * data['yield_per_hectare']
            predicted_price = calculated_yield * data['price_per_tonne']

            return render_template('yield_result.html',
                                   crop=crop_type,
                                   area=f"{area} {unit}",
                                   yield_val=f"{calculated_yield:.2f} Tonnes",
                                   price=f"₹ {predicted_price:,.2f}")
        else:
            return "Selected crop is not available for prediction.", 400
    except Exception as e:
        return f"An error occurred during calculation: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)

