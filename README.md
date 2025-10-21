# CropTellect: AI Crop Disease Detector

[](https://www.python.org/downloads/)
[](https://flask.palletsprojects.com/)
[](https://www.tensorflow.org/)
[](LICENSE)

> *“The future of farming isn’t in the soil — it’s in the data.”*

An AI-powered web application that detects crop diseases using deep learning. Built with **TensorFlow**, **Flask**, and a fine-tuned **VGG19 model**, this project aims to support farmers and agricultural experts with quick, reliable disease detection directly from leaf images.

-----

## Table of Contents

  * [Features](#features)
  * [Tech Stack](#tech-stack)
  * [How It Works](#how-it-works)
  * [Project Structure](#project-structure)
  * [Dataset](#dataset)
  * [Getting Started](#getting-started)
      * [Prerequisites](#prerequisites)
      * [Installation](#installation)
  * [Usage](#usage)
  * [Future Enhancements](#future-enhancements)
  * [Contributing](#contributing)
  * [License](#license)

-----

## Features

  *  Deep Learning Model (VGG19) trained on thousands of diseased and healthy crop images
  *  Detects multiple crop diseases with high accuracy
  *  Clean and responsive web interface (Flask + HTML/CSS)
  *  Real-time predictions from uploaded images
  *  Modular and scalable architecture for future model updates

-----

## Tech Stack

| Component | Technologies Used |
| :--- | :--- |
| **Frontend** | HTML, CSS, Flask Templates |
| **Backend** | Python (Flask Framework) |
| **Model** | TensorFlow / Keras (VGG19 Architecture) |
| **Deployment** | Localhost / Future Cloud Hosting |
| **Dataset** | [Kaggle Crop Disease Dataset](https://www.kaggle.com/datasets/atharvpose/crops-disease) |

-----

## How It Works

1.  **Upload:** A user uploads a crop leaf image via the web interface.
2.  **Preprocessing:** The backend preprocesses the image to match the model's required input dimensions (e.g., $224 \times 224 \times 3$).
3.  **Model Prediction:** The image is passed to the loaded VGG19 TFLite model, which classifies the disease.
4.  **Output Displayed:** The prediction (disease name) is sent back to the user and displayed on the results page.

-----

## Project Structure

```
CropTellect/
│
├── app/
│   ├── app.py                # Main Flask application logic
│   ├── static/
│   │   └── css/style.css     # For CSS, images, and JS files
│   ├── templates/
│   │   ├── index.html        # Upload page
│   │   └── result.html       # Prediction result page
│   └── VGG19_model.tflite    # Trained model (excluded via .gitignore)
│
├── Notebook/
│   └── crop-disease-detector.ipynb  # Model training and experimentation
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored large files and dataset
└── README.md
```

-----

## Dataset

The model was trained on a curated dataset available on Kaggle, containing thousands of images of healthy and diseased crop leaves.

 **Kaggle Dataset:** [Crop Disease Dataset (Atharv Pose)](https://www.kaggle.com/datasets/atharvpose/crops-disease)

> *“Every pixel has a story; the model just needs to listen.”*

-----

## Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

  * Python 3.9+
  * `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/Atharv3142M/CropTellect.git
    cd CropTellect
    ```

2.  **Create and activate a virtual environment:**

    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4.  **Add the model file:**

      * Download or train your `VGG19_model.tflite` file.
      * Place it inside the `app/` directory.

-----

## Usage

1.  **Run the Flask server:**

    ```sh
    python app/app.py
    ```

2.  **Open the application:**

      * Open your web browser and navigate to `http://127.0.0.1:5000`.

3.  **Get a prediction:**

      * Click "Choose File" to upload an image of a crop leaf.
      * Press the "Predict" button to see the diagnosis.

-----

## Future Enhancements

  * Integration with **live camera feeds** for real-time diagnosis
  * Expand to detect **more crop species** and disease types
  * Deploy on **Streamlit, AWS, Heroku, or Hugging Face Spaces**
  * Add **disease treatment recommendations** based on the prediction
  * Develop a **REST API** endpoint for programmatic access

-----

## Contributing

Contributions are what make the open-source community such an amazing place. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## License

Distributed under the MIT License. See `LICENSE` file for more information.