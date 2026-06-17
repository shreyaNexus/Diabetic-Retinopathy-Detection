📖 Diabetic Retinopathy Detection:

🩺 Overview:
Diabetic Retinopathy (DR) is a diabetes-related eye disease that damages the retina and can lead to permanent vision loss if not diagnosed early. Manual screening of retinal fundus images requires experienced ophthalmologists and can be time-consuming, especially in regions with limited healthcare resources.
This project develops an AI-powered Diabetic Retinopathy Detection System that automatically analyzes retinal fundus images and classifies the severity of the disease. The system combines advanced image enhancement techniques such as CLAHE and SwinIR with state-of-the-art deep learning models (EfficientNet, DenseNet, and MobileNet) to achieve accurate diagnosis. Additionally, Grad-CAM is used to provide visual explanations, making the model's predictions more interpretable and trustworthy.

🎯 Objectives:
Detect Diabetic Retinopathy from retinal fundus images.
Classify retinal images into five severity levels.
Improve image quality using CLAHE and SwinIR.
Compare the performance of EfficientNet, DenseNet, and MobileNet.
Provide explainable predictions using Grad-CAM heatmaps.
Assist healthcare professionals in early diagnosis and treatment planning.
Reduce manual screening workload through automation.

✨ Key Features:
🔍 Automated DR Detection

Detects diabetic retinopathy from retinal fundus images with minimal human intervention.

🖼️ Advanced Image Enhancement

Uses CLAHE and SwinIR to improve image quality and highlight retinal abnormalities.

🤖 Multiple Deep Learning Models
EfficientNet
DenseNet
MobileNet
📊 Severity Classification

Classifies retinal images into:

No DR
Mild DR
Moderate DR
Severe DR
Proliferative DR
🔥 Explainable AI

Uses Grad-CAM to visualize disease-affected retinal regions.

🌐 User-Friendly Interface
Allows users to upload images and receive instant predictions.
📈 Performance Analytics
Displays accuracy, precision, recall, F1-score, and confusion matrix.

🛠️ Tech Stack:
Category	Technologies
Programming Language	Python
Deep Learning	TensorFlow, Keras, PyTorch
Computer Vision	OpenCV, Pillow
Data Processing	NumPy, Pandas
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Explainable AI	Grad-CAM
Image Enhancement	CLAHE, SwinIR
Web Framework	Flask / Streamlit
Version Control	Git, GitHub

💡 Unique Selling Points (USP):
1. Dual Image Enhancement Pipeline
Combines CLAHE and SwinIR for superior retinal image quality.

2. Multi-Model Comparison
Evaluates EfficientNet, DenseNet, and MobileNet to identify the best-performing architecture.

3. Explainable Predictions
Grad-CAM heatmaps provide transparency and interpretability.

4. Lightweight Deployment Option
MobileNet enables deployment on low-resource devices and mobile platforms.

5. Healthcare-Focused AI Solution
Designed specifically for ophthalmology and diabetic screening programs.

6. Scalable Architecture
Can be extended to detect other retinal diseases such as glaucoma and macular degeneration.

📈 Business Impact & Insights:
Healthcare Benefits
Enables early detection of diabetic retinopathy.
Reduces risk of blindness through timely intervention.
Supports ophthalmologists in diagnosis.
Operational Benefits
Reduces manual screening time.
Improves diagnostic efficiency.
Handles large-scale patient screening.
Economic Benefits
Lowers healthcare costs associated with advanced-stage treatment.
Increases accessibility of eye care in rural and underserved areas.
AI-Driven Insights
Identifies disease severity automatically.
Highlights affected retinal regions.
Generates interpretable diagnostic reports.

📂 Data Sources:
Primary Datasets:
 EyePACS Dataset
    Large-scale retinal image dataset.
    Widely used in diabetic retinopathy research.
 APTOS 2019 Blindness Detection Dataset
    High-quality retinal fundus images.
    Annotated with DR severity levels.
Data Classes
Label   	Class
0	        No DR
1       	Mild DR
2	        Moderate DR
3	        Severe DR
4        	Proliferative DR

🚀 Future Improvements:

Ensemble Learning
Combine EfficientNet, DenseNet, and MobileNet predictions for improved accuracy.
Vision Transformers
Integrate Vision Transformer (ViT) architectures.
Mobile Application
Develop Android and iOS applications for portable screening.
Cloud Deployment
Deploy on AWS, Azure, or Google Cloud for remote diagnosis.
Multi-Disease Detection
Extend the system to detect:
Glaucoma
Cataracts
Age-related Macular Degeneration (AMD)
Real-Time Screening
Enable real-time retinal image analysis in hospitals and clinics.
Electronic Health Record Integration
Connect predictions with hospital management systems.

Screnshots:

![Dashboard](https://github.com/shreyaNexus/Diabetic-Retinopathy-Detection/blob/main/Screenshot%202026-06-17%20201754.png?raw=true)

