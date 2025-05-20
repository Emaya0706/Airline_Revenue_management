# Airline_Revenue_management

# Airline Revenue Management Predictive Analytics

A predictive analytics framework for Airline Revenue Management (ARM) that leverages supervised machine learning techniques to forecast ticket prices and simulate revenue outcomes based on historical and operational flight data.
---

## 📌 Project Objectives

1. **Price Prediction**  
   - Train regression models to forecast airline ticket prices using historical booking, flight, and market variables.  

2. **Revenue Simulation**  
   - Simulate total and monthly revenues by combining predicted prices with seat-demand data.  

3. **Model Comparison**  
   - Evaluate and compare the performance of Random Forest, XGBoost, and LSTM models.  

4. **Actionable Insights**  
   - Identify optimal pricing strategies and peak revenue periods to inform dynamic pricing decisions.

---

## 📊 Dataset

The dataset contains **98,619** records and **24** features covering:

| Category                    | Example Features                              |
|-----------------------------|-----------------------------------------------|
| **Passenger Demographics**  | Age, Gender, Nationality                      |
| **Booking Characteristics** | Booking_Lead_Time, Fare_Class                 |
| **Flight Details**          | Seats_Available, Seats_Booked, Public_Holiday |
| **Market Data**             | Competitor_Price, Fuel_Price                  |

All preprocessing and feature engineering steps are documented in the dissertation (Appendix B).

---

## 🗂️ Folder Structure

Airline_Revenue_Management/

├── .conda/                    
├── data/                      
│   ├── train_features.csv     
│   ├── train_labels.csv       
│   ├── test_features.csv      
│   └── test_labels.csv        

├── dataset/                   
│   └── Airline_Revenue_Management_Dataset.xlsx

├── output_images/             
│   ├── revenue_histogram.png  
│   ├── residuals_rf.png       
│   └── qqplot_lstm.png 
├── scripts/                   
│   └── ticket_price_prediction.py

├── README.md                  
└── requirements.txt           


---

## 🧠 Models & Evaluation

| Model                    | RMSE   | MAE    | R²     | MAPE  |
|--------------------------|--------|--------|--------|-------|
| Random Forest Regressor  | 68.12  | 42.37  | 0.958  | 7.2%  |
| XGBoost Regressor        | **55.23**  | **35.89**  | **0.973**  | **5.8%**  |
| LSTM Neural Network      | 62.87  | 39.14  | 0.965  | 6.5%  |

- **Best model:** XGBoost (lowest RMSE, highest R²)

---

## 📈 Key Findings

- **Simulated Total Revenue:** £1.86 billion  
- **Peak Revenue Months:** July & December  
- **Insight:** Dynamic fare adjustments prior to peak‑season booking lead times yield the highest revenue uplift.
---

## 🛠️ Dependencies

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- xgboost  
- tensorflow / keras  



