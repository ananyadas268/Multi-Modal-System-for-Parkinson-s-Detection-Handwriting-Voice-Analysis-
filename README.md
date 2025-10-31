🧠 Multi-Modal AI System for Parkinson's Disease Detection

📖 Project Overview  
This project presents a web-based system designed for early-stage screening of Parkinson's Disease (PD) using advanced **Deep Learning models**.  
It is a multi-modal solution**, leveraging two non-invasive biomarkers:  
- ✍️ Handwriting motor control analysis
- 🎤 Voice acoustic impairment detection

The application is built using Streamlit , providing an interactive interface that allows users to upload data and instantly receive predictive scores.

---

🎯 Core Functionality  

1. ✍️ Handwriting Analysis (Motor Control)
This module evaluates **micro-motor impairments** linked to PD, such as **micrographia** or **dysgraphia**.  

Models Used:
- Spiral Model – Custom CNN trained on spiral drawing samples  
  *Validation Accuracy:* ~78.91%  
- Wave Model – VGG16 (Transfer Learning) trained on wave drawing samples  
  Validation Accuracy: ~81.25%

Input Data:
- Two image files (`.jpg` / `.png`):  
  - One *Spiral Drawing*  
  - One *Wave Drawing*

**Prediction Logic:**
- If either the Spiral or Wave model’s prediction score exceeds the **0.5 threshold**, the combined handwriting diagnosis is **marked positive** (OR logic).

---

2. 🎤 Voice Analysis (Hybrid Acoustic Model)
This module identifies **early speech abnormalities (dysarthria)** by analyzing voice patterns such as **pitch, amplitude,** and **noise variations**.

Model Used:
- Hybrid CNN Acoustic Model**  
  *Validation Accuracy:* ~91.50%
Input Data:
- One acoustic file (`.wav`) of a **sustained vowel sound** (e.g., “Ahhh”)  
- Four clinical/acoustic parameters:  
  - Age
  - Jitter 
  - Shimmer  
  - HNR (Harmonic-to-Noise Ratio)

Note:
The extraction of the **2048-dimensional acoustic feature vector** from `.wav` files is currently **simulated** using randomized data for demonstration.  
In a production environment, this would be replaced by genuine feature extraction using libraries such as **Librosa** or **OpenSMILE**.

---

# 🧠 Multi-Modal AI System for Parkinson's Disease Detection

## 📖 Project Overview  
This project presents a **web-based system** designed for **early-stage screening of Parkinson's Disease (PD)** using advanced **Deep Learning models**.  
It is a **multi-modal solution**, leveraging two non-invasive biomarkers:  
- ✍️ **Handwriting motor control analysis**  
- 🎤 **Voice acoustic impairment detection**

The application is built using **Streamlit**, providing an interactive interface that allows users to upload data and instantly receive predictive scores.

---

## 🎯 Core Functionality  

### 1. ✍️ Handwriting Analysis (Motor Control)
This module evaluates **micro-motor impairments** linked to PD, such as **micrographia** or **dysgraphia**.  

**Models Used:**
- **Spiral Model** – Custom CNN trained on spiral drawing samples  
  *Validation Accuracy:* ~78.91%  
- **Wave Model** – VGG16 (Transfer Learning) trained on wave drawing samples  
  *Validation Accuracy:* ~81.25%

**Input Data:**
- Two image files (`.jpg` / `.png`):  
  - One *Spiral Drawing*  
  - One *Wave Drawing*

**Prediction Logic:**
- If either the Spiral or Wave model’s prediction score exceeds the **0.5 threshold**, the combined handwriting diagnosis is **marked positive** (OR logic).

---

### 2. 🎤 Voice Analysis (Hybrid Acoustic Model)
This module identifies **early speech abnormalities (dysarthria)** by analyzing voice patterns such as **pitch, amplitude,** and **noise variations**.

**Model Used:**
- **Hybrid CNN Acoustic Model**  
  *Validation Accuracy:* ~91.50%

**Input Data:**
- One acoustic file (`.wav`) of a **sustained vowel sound** (e.g., “Ahhh”)  
- Four clinical/acoustic parameters:  
  - **Age**  
  - **Jitter**  
  - **Shimmer**  
  - **HNR (Harmonic-to-Noise Ratio)**

**Note:**  
The extraction of the **2048-dimensional acoustic feature vector** from `.wav` files is currently **simulated** using randomized data for demonstration.  
In a production environment, this would be replaced by genuine feature extraction using libraries such as **Librosa** or **OpenSMILE**.

---

## 🛠️ Technology Stack  
| Category | Tools & Libraries |
|-----------|------------------|
| **Web Framework** | Streamlit |
| **Machine Learning** | TensorFlow / Keras |
| **Data Processing** | Python, NumPy, Pandas, PIL (Pillow) |
| **Dataset** | [Sowmya Barla / Parkinson’s Augmented Handwriting Dataset (Kaggle)](https://www.kaggle.com/datasets/sowmyabarla/parkinsons-augmented-handwriting-dataset) |

---

## 🚀 Installation & Usage  

### Prerequisites  
Ensure you have **Python 3.8+** installed.

### Steps to Run
1. **Clone the repository** or download the project files:
   ```bash
   git clone https://github.com/<your-username>/parkinsons-multimodal-ai.git
   cd parkinsons-multimodal-ai


---

🚀 Installation & Usage  

 Prerequisites  
Ensure you have Python 3.8+ installed.

Steps to Run
1.Clone the repository or download the project files:
   ```bash
   git clone https://github.com/<your-username>/parkinsons-multimodal-ai.git
   cd parkinsons-multimodal-ai

2. Place the models in the same directory as app.py
    spiral_model_78_91_acc.keras
    wave_model_81_25_acc.keras
    hybrid_cnn_acoustic_model(2).keras

3. Install dependencies:
    pip install streamlit tensorflow numpy pandas pillow

4. streamlit run app.py

⚠️ Medical Disclaimer

This application is intended solely for research and screening purposes.
It is not a substitute for professional medical diagnosis, advice, or treatment.
Always consult a qualified healthcare provider with any medical concerns or questions.
🧩 Future Improvements

Integration of real audio feature extraction (Librosa/OpenSMILE)

Deployment on Streamlit Cloud or Hugging Face Spaces

Addition of Explainable AI (XAI) components for interpretability

Real-time data collection for continuous model retraining

👩‍💻 Author

Ananya Das
📧 ananyadas268@gmail.com

