🤖 AI/ML Engineering Internship Portfolio
Organization: DevelopersHub Corporation
Intern: Muhammad Numan
Timeline: April 2026
📌 Overview
This repository contains the completed tasks for the AI/ML Engineering Internship. The projects cover the full spectrum of the machine learning pipeline: from Exploratory Data Analysis (EDA) and data visualization to training advanced regression models and fine-tuning Large Language Models (LLMs).
🚀 Tasks Completed:
Task 5: Mental Health Support Chatbot (Fine-Tuned LLM)
Task 6: House Price Prediction (Gradient Boosting & Streamlit)
Task 1: Iris Dataset Exploration & Visualization
🌿 Task 5: Serenity — Mental Health Support Chatbot
Objective: Build a supportive, empathetic AI companion fine-tuned on real human emotional conversations to provide comfort for stress and anxiety.
✨ Key Features
Base Model: DistilGPT2 (82M Parameters).
Dataset: Facebook AI’s EmpatheticDialogues (~25k samples).
UI/UX: A clean, calming web interface built with Streamlit and deployed via Netlify.
Functionality: Includes emotion-state selection (Sad, Anxious, etc.) to guide the conversation.
Landing Page	Chat Interface
![alt text](/Screenshots/image-19%20(1).png)
![alt text](/Screenshots/image-19%20(3).png)
![alt text](/Screenshots/image-19%20(2).png)
HuggingFace:
![alt text](/Screenshots/image-18.png)

🛠️ Training & Tech Stack
The model was fine-tuned using the Hugging Face Trainer API. Below are the training metrics and the architecture overview.
![alt text](/Screenshots/image-15.png)


Live App (Static): Serenity on Netlify
Model Space: Health.ai on Hugging Face
🏠 Task 6: House Price Prediction Dashboard
Objective: Predict California housing prices using property features like median income, house age, and location.
📊 Model Performance
I implemented a Gradient Boosting Regressor, achieving high accuracy.
R-Squared Score: ~0.81
Mean Absolute Error: ~$0.33 (in units of $100k)
🖥️ Interactive Streamlit Dashboard
The application features a professional dark-mode UI with three main sections:
EDA: Live dataset preview and correlation heatmaps.
Metrics: Real-time model performance visualization.
Prediction Tool: Interactive sliders for users to estimate market value instantly.
![alt text](/Screenshots/image-9.png)

![alt text](/Screenshots/image-11.png)
🌸 Task 1: Iris Dataset Visualization
Objective: Perform comprehensive Exploratory Data Analysis (EDA) on the classic Iris dataset to identify patterns and outliers.
🔍 Insights & Visuals
Statistical Summary: Used .describe() and .info() for data integrity checks.
Relationships: Utilized Seaborn pairplot and scatterplot to visualize cluster separation between species.
Distributions: Histograms and Box plots were used to detect outliers and feature spread.
Feature Relationships	Data Distribution
![alt text](/Screenshots/image-5.png)
![alt text](/Screenshots/image-7.png)
🛠️ Technologies Used
Languages: Python
Data Science: Pandas, NumPy, Scikit-Learn
Visualization: Matplotlib, Seaborn
Deep Learning: Hugging Face Transformers, Trainer API, PyTorch
Deployment: Streamlit, Netlify, Hugging Face Spaces
⚙️ How to Run Locally
1. Clone the Repository
code
Bash
git clone https://github.com/NumanTahir/Internship_ML.git
cd Internship_ML
2. Install Dependencies
code
Bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit transformers torch
3. Launch the Applications
For House Price App: streamlit run App.py
For Mental Health Chatbot: streamlit run Serenity_App.py
