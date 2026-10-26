import joblib, pandas as pd
import preprocessing  # noqa: F401  (joblib needs your custom classes importable)

model = joblib.load("outputs/car_price_pipeline.joblib")
car = pd.DataFrame([{
    "Make": "Hyundai", "Model": "i20 Sportz 1.2", "Year": 2018, "Kilometer": 45000,
    "Fuel Type": "Petrol", "Transmission": "Manual", "Location": "Mumbai", "Color": "White",
    "Owner": "First", "Seller Type": "Individual", "Engine": "1197 cc", "Max Power": "81 bhp @ 6000 rpm",
    "Max Torque": "115 Nm @ 4000 rpm", "Drivetrain": "FWD", "Length": 3995, "Width": 1734,
    "Height": 1505, "Seating Capacity": 5, "Fuel Tank Capacity": 45}])
print(f"Estimated price: {model.predict(car)[0]:,.0f}")
