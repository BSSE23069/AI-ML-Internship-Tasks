                              AI/ML Engineering Internship Tasks
                             Organization: DevelopersHub Corporation
                                  Intern:Muhammad Numan
Tasks Completed: Task 1 & Task 6
                                 🛠️ Environment Setup
Before running the tasks, ensure you have the required libraries installed:
                                      code
Bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit
                           💊 Task 5: Mentel_health_chatboard

website:
       you can click link to see it 
       "https://florentine-numan.netlify.app/"
 ![alt text](/Screenshots/image-13.png) 
 ![alt text](/Screenshots/image-14.png)     
 ![alt text](/Screenshots/image-15.png)
 ![alt text](/Screenshots/image-16.png)
 ![alt text](/Screenshots/image-17.png)

                         🌸 Task 1: Exploring and Visualizing a Simple Dataset

Objective: Load, inspect, and visualize the Iris Dataset to understand data trends and feature distributions.

                                      📂 Files
Task1_Iris_Visualization.ipynb: Jupyter Notebook containing data analysis and visualization.
                                  🚀 How to Run
Open the .ipynb file in VS Code or Jupyter Notebook.

Run all cells to see the data summary and generated plots.
                               📊 Visual Insights
Step	Description	Screenshot
Data Inspection	Using .head(), .info(), and .describe() to understand the structure.	
![alt text](/Screenshots/image-2.png)
Feature Relationships	Scatter plots showing sepal and petal dimensions.	
![alt text](/Screenshots/image-3.png)
![alt text](/Screenshots/image-4.png)
Global View	A Pairplot visualizing relationships between all features simultaneously.	
![alt text](/Screenshots/image-5.png)
Distributions	Histograms showing the frequency and range of feature values.	
![alt text](/Screenshots/image-6.png)
Outlier Detection	Box plots used to identify data spread and potential outliers per species.	
![alt text](/Screenshots/image-7.png)
                          

                        🏠 Task 6: House Price Prediction
Objective: Predict California house prices using property features (income, age, location) with a Gradient Boosting model and a Streamlit dashboard.
                                📂 Files
House_Price_Prediction.py: Backend script for model training and console evaluation.
App.py: Full-stack Streamlit application featuring EDA and a Predictive UI.
                        🚀 How to Run
1. Terminal Script:
open:
python House_Price_Prediction.ipynb
run each cell 
2. Interactive Web App:
code
Bash
streamlit run App.py
📈 Model & UI Results
Backend Performance
The model utilizes a Gradient Boosting Regressor. The console output confirms the training success and provides evaluation metrics.
R-Squared Score: ~0.81 (High accuracy).
Mean Absolute Error (MAE): Tracking average prediction deviation.
![alt text](/Screenshots/image-8.png)
Interactive Dashboard (Streamlit)
The web application is divided into three professional sections:
Exploratory Data Analysis (EDA): Displays a preview of the raw census data and a Correlation Heatmap to show which features (like Median Income) most impact house prices.
![alt text](/Screenshots/image-9.png)
Model Performance Metrics: Real-time display of Accuracy and Error scores in a dashboard format.
![alt text](/Screenshots/image-10.png)
Predictive Tool: An interface where users can input custom data to receive an instant price estimation.
![alt text](/Screenshots/image-11.png)

![alt text](/Screenshots/image-12.png)
🛠️ Technologies Used
Python: Core logic.
Pandas/NumPy: Data manipulation.
Matplotlib/Seaborn: Data visualization.
Scikit-Learn: Machine Learning (Gradient Boosting).
Streamlit: Web deployment and UI.
