from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Landslide Prediction API")

# 1. CORS Middleware: Allows the Streamlit UI to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the locked-in Machine Learning artifacts
try:
    model = joblib.load("landslide_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading ML artifacts: {e}")

# 3. Strict Data Validation (10 features)
class SensorPayload(BaseModel):
    Soil_Saturation: float = Field(..., ge=0.0, le=1.0)
    Slope_Angle: float = Field(..., ge=0.0, le=90.0)
    Rainfall_72h_mm: float = Field(..., ge=0.0)
    Rainfall_mm: float = Field(..., ge=0.0)
    Earthquake_Activity: float = Field(..., ge=0.0)
    Lithology_Weakness: float = Field(..., ge=0.0, le=1.0)
    Surface_Crack_Width_cm: float = Field(..., ge=0.0)
    Hill_Cutting_Severity: float = Field(..., ge=0.0, le=1.0)
    Distance_to_Road_m: float = Field(..., ge=0.0)
    Historical_Failure_Count: int = Field(..., ge=0)

# 4. Root and Health Check Endpoints
@app.get("/")
def read_root():
    return {"message": "Landslide Prediction API is active. Navigate to /docs to test endpoints."}

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": True if model and scaler else False
    }

# 5. The Prediction Endpoint with 50 Deterministic Guardrails
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Hybrid Landslide Prediction & Expert System")

# 1. CORS Middleware: Allows the Streamlit UI to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the locked-in Machine Learning artifacts
try:
    model = joblib.load("landslide_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading ML artifacts: {e}")

# 3. Strict Data Validation (10 features matching training order)
class SensorPayload(BaseModel):
    Soil_Saturation: float = Field(..., ge=0.0, le=1.0)
    Slope_Angle: float = Field(..., ge=0.0, le=90.0)
    Rainfall_72h_mm: float = Field(..., ge=0.0)
    Rainfall_mm: float = Field(..., ge=0.0)
    Earthquake_Activity: float = Field(..., ge=0.0)
    Lithology_Weakness: float = Field(..., ge=0.0, le=1.0)
    Surface_Crack_Width_cm: float = Field(..., ge=0.0)
    Hill_Cutting_Severity: float = Field(..., ge=0.0, le=1.0)
    Distance_to_Road_m: float = Field(..., ge=0.0)
    Historical_Failure_Count: int = Field(..., ge=0)

# 4. Root and Health Check Endpoints
@app.get("/")
def read_root():
    return {"message": "Hybrid Landslide Prediction API is active. Navigate to /docs to test endpoints."}

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": True if model and scaler else False
    }

# 5. The Hybrid Prediction Endpoint
@app.post("/predict")
def predict_landslide(data: SensorPayload):
    try:
        # 1. Prepare features in the exact training order
        features = np.array([[
            data.Rainfall_mm,
            data.Slope_Angle,
            data.Soil_Saturation,
            data.Earthquake_Activity,
            data.Rainfall_72h_mm,
            data.Distance_to_Road_m,
            data.Lithology_Weakness,
            data.Historical_Failure_Count,
            data.Hill_Cutting_Severity,
            data.Surface_Crack_Width_cm
        ]])

        # 2. Base Machine Learning Inference (For normal/standard conditions)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        prob_pct = round(probability * 100, 2)
        
        if prob_pct >= 75.0:
            alert_status = "DANGER 🔴"
        elif prob_pct >= 55.0:
            alert_status = "WARNING 🟠"
        else:
            alert_status = "SAFE 🟢"

        # 3. OVERRIDE LAYER: 50 DETERMINISTIC GEOTECHNICAL GUARDRAILS
        boundary_exceeded = False

        # 1. Brittle Rockfall (Dry, Weak, Steep)
        if data.Slope_Angle >= 80.0 and data.Lithology_Weakness >= 0.90 and data.Soil_Saturation <= 0.30:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Brittle Rockfall on Degraded Cliff Face"
            prob_pct = 95.0
            boundary_exceeded = True

        # 2. Reactivation of Historical Slide
        elif data.Historical_Failure_Count >= 3 and (data.Rainfall_mm >= 50.0 or data.Earthquake_Activity >= 3.0):
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Reactivation of Historical Shear Zone"
            prob_pct = 92.0
            boundary_exceeded = True

        # 3. Critical Hydrostatic Stress
        elif data.Slope_Angle >= 75.0 and data.Soil_Saturation >= 0.85:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Hydrostatic Shear Failure on High-Gradient Slope"
            prob_pct = 90.0
            boundary_exceeded = True

        # 4. Hydrostatic Wedging
        elif data.Surface_Crack_Width_cm >= 5.0 and data.Soil_Saturation >= 0.90:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Hydrostatic Wedging and Fissure Expansion"
            prob_pct = 94.0
            boundary_exceeded = True

        # 5. Progressive Slump / Creep
        elif data.Surface_Crack_Width_cm >= 4.0 and data.Soil_Saturation >= 0.85 and data.Slope_Angle >= 45.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Progressive Slump and Impending Liquefaction"
            prob_pct = 88.0
            boundary_exceeded = True

        # 6. Critical Toe Undercutting
        elif data.Distance_to_Road_m <= 5.0 and data.Hill_Cutting_Severity >= 0.90 and data.Slope_Angle >= 60.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Basal Undercutting Collapse Near Roadway"
            prob_pct = 91.0
            boundary_exceeded = True

        # 7. Highway Excavation Washout
        elif data.Hill_Cutting_Severity >= 0.85 and data.Distance_to_Road_m <= 10.0 and data.Rainfall_mm >= 100.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Extreme Highway Excavation Washout"
            prob_pct = 89.0
            boundary_exceeded = True

        # 8. Seismic Liquefaction
        elif data.Earthquake_Activity >= 4.0 and data.Soil_Saturation >= 0.80:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Seismic Liquefaction of Saturated Soil"
            prob_pct = 96.0
            boundary_exceeded = True

        # 9. Catastrophic Washout
        elif data.Rainfall_72h_mm >= 300.0 or data.Rainfall_mm >= 150.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Catastrophic Flash Flood Washout"
            prob_pct = 98.0
            boundary_exceeded = True
            
        # 10. Severe Kinetic Shear
        elif data.Earthquake_Activity >= 6.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Massive Seismic Shear Plane Failure"
            prob_pct = 99.0
            boundary_exceeded = True

        # 11. Artificial Excavation Collapse
        elif data.Lithology_Weakness >= 0.85 and data.Hill_Cutting_Severity >= 0.85:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Anthropogenic Excavation Failure in Weak Strata"
            prob_pct = 87.0
            boundary_exceeded = True

        # 12. Massive Structural Tearing
        elif data.Surface_Crack_Width_cm >= 10.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Advanced Tension Crack Disconnect"
            prob_pct = 93.0
            boundary_exceeded = True

        # 13. Post-Seismic Hydrostatic Lubrication
        elif data.Earthquake_Activity >= 3.5 and data.Rainfall_mm >= 80.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Post-Seismic Hydrostatic Lubrication"
            prob_pct = 91.0
            boundary_exceeded = True

        # 14. Basal Drainage Blowout
        elif data.Distance_to_Road_m <= 3.0 and data.Soil_Saturation >= 0.95 and data.Slope_Angle >= 45.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Basal Saturation and Toe Blowout"
            prob_pct = 94.0
            boundary_exceeded = True

        # 15. Spontaneous Geologic Disintegration
        elif data.Lithology_Weakness >= 0.95 and data.Surface_Crack_Width_cm >= 8.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Spontaneous Geologic Disintegration"
            prob_pct = 95.0
            boundary_exceeded = True

        # 16. Low-Frequency Vibration Collapse
        elif data.Distance_to_Road_m <= 2.0 and data.Hill_Cutting_Severity >= 0.85 and data.Historical_Failure_Count >= 1 and data.Earthquake_Activity >= 2.5:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Low-Frequency Vibration Collapse on Fractured Cut"
            prob_pct = 89.0
            boundary_exceeded = True

        # 17. Deep-Seated Rotational Shear Failure
        elif data.Rainfall_72h_mm >= 250.0 and data.Soil_Saturation >= 0.90 and data.Lithology_Weakness >= 0.65 and data.Slope_Angle >= 35.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Deep-Seated Rotational Shear Plane Detachment"
            prob_pct = 92.0
            boundary_exceeded = True

        # 18. Cohesionless Seismic Shatter
        elif data.Slope_Angle >= 60.0 and data.Lithology_Weakness >= 0.85 and data.Earthquake_Activity >= 4.5:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Cohesionless Seismic Shatter of Brittle Strata"
            prob_pct = 96.0
            boundary_exceeded = True

        # 19. Internal Subsurface Piping Blowout
        elif data.Rainfall_72h_mm >= 350.0 and data.Lithology_Weakness >= 0.75 and data.Slope_Angle >= 50.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Internal Subsurface Piping and Basal Blowout"
            prob_pct = 97.0
            boundary_exceeded = True

        # 20. Accelerated Tension Rupture
        elif data.Surface_Crack_Width_cm >= 6.0 and data.Historical_Failure_Count >= 2 and data.Slope_Angle >= 55.0 and data.Lithology_Weakness >= 0.70:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Accelerated Tension Rupture of Unstable Shear Plane"
            prob_pct = 90.0
            boundary_exceeded = True

        # 21. Delayed Post-Storm Pore Pressure Liquefaction
        elif data.Rainfall_mm <= 10.0 and data.Rainfall_72h_mm >= 200.0 and data.Soil_Saturation >= 0.85:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Delayed Post-Storm Pore Pressure Liquefaction"
            prob_pct = 86.0
            boundary_exceeded = True

        # 22. Catastrophic Oversteepening of Highway Cut
        elif data.Hill_Cutting_Severity >= 0.95 and data.Distance_to_Road_m <= 1.0 and data.Slope_Angle >= 85.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Catastrophic Oversteepening of Unbuttressed Highway Cut"
            prob_pct = 98.0
            boundary_exceeded = True

        # 23. Seismically Induced Tension Fracture Propagation
        elif data.Surface_Crack_Width_cm >= 4.0 and data.Earthquake_Activity >= 3.5 and data.Slope_Angle >= 50.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Seismically Induced Tension Fracture Propagation"
            prob_pct = 89.0
            boundary_exceeded = True

        # 24. Rain-Induced Brittle Rock Avalanche
        elif data.Slope_Angle >= 75.0 and data.Lithology_Weakness >= 0.80 and data.Rainfall_mm >= 120.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Rain-Induced Brittle Rock Avalanche"
            prob_pct = 94.0
            boundary_exceeded = True

        # 25. Basal Subgrade Liquefaction and Toe Subsidence
        elif data.Distance_to_Road_m <= 2.0 and data.Soil_Saturation >= 0.95 and data.Historical_Failure_Count >= 2:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Basal Subgrade Liquefaction and Toe Subsidence"
            prob_pct = 92.0
            boundary_exceeded = True

        # 26. Terminal Acceleration of Chronic Creep Zone
        elif data.Historical_Failure_Count >= 4 and data.Surface_Crack_Width_cm >= 7.0 and data.Slope_Angle >= 40.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Terminal Acceleration of Chronic Creep Zone"
            prob_pct = 91.0
            boundary_exceeded = True

        # 27. Vibration-Induced Shear on Unbuttressed Excavation
        elif data.Hill_Cutting_Severity >= 0.90 and data.Distance_to_Road_m <= 3.0 and data.Earthquake_Activity >= 2.0 and data.Lithology_Weakness >= 0.70:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Acoustic/Vibration-Induced Shear on Unbuttressed Cut"
            prob_pct = 88.0
            boundary_exceeded = True

        # 28. Fracture-Flow Hydrostatic Blowout
        elif data.Lithology_Weakness >= 0.85 and data.Surface_Crack_Width_cm >= 5.0 and data.Rainfall_mm >= 100.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Deep Fracture-Flow Hydrostatic Blowout"
            prob_pct = 93.0
            boundary_exceeded = True

        # 29. Multi-Hazard Synergistic Collapse
        elif data.Soil_Saturation >= 0.85 and data.Earthquake_Activity >= 5.0 and data.Rainfall_mm >= 80.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Multi-Hazard Synergistic Topographical Collapse"
            prob_pct = 99.0
            boundary_exceeded = True

        # 30. A-Hydrous Gravitational Mass Subsidence
        elif data.Rainfall_72h_mm <= 10.0 and data.Surface_Crack_Width_cm >= 12.0 and data.Lithology_Weakness >= 0.90:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: A-Hydrous Gravitational Mass Subsidence"
            prob_pct = 90.0
            boundary_exceeded = True

        # 31. Hydrologically Accelerated Toe Scour and Subsidence
        elif data.Distance_to_Road_m <= 3.0 and data.Rainfall_72h_mm >= 250.0 and data.Slope_Angle >= 45.0 and data.Soil_Saturation >= 0.85:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Hydrologically Accelerated Toe Scour and Subsidence"
            prob_pct = 91.0
            boundary_exceeded = True

        # 32. Kinetic Reactivation of Paleolandslide Shear Plane
        elif data.Historical_Failure_Count >= 5 and data.Earthquake_Activity >= 3.0 and data.Lithology_Weakness >= 0.60:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Kinetic Reactivation of Paleolandslide Shear Plane"
            prob_pct = 94.0
            boundary_exceeded = True

        # 33. Rapid Drawdown Pore Pressure Imbalance
        elif data.Rainfall_72h_mm >= 300.0 and data.Rainfall_mm <= 5.0 and data.Soil_Saturation >= 0.95 and data.Slope_Angle >= 50.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Rapid Drawdown Pore Pressure Imbalance"
            prob_pct = 95.0
            boundary_exceeded = True

        # 34. Micro-Lubrication on Unretained Roadcut
        elif data.Hill_Cutting_Severity >= 0.95 and data.Lithology_Weakness >= 0.80 and data.Rainfall_mm >= 20.0 and data.Distance_to_Road_m <= 5.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Frictional Loss via Micro-Lubrication on Unretained Cut"
            prob_pct = 87.0
            boundary_exceeded = True

        # 35. Terminal Aseismic Gravitational Shearing
        elif data.Surface_Crack_Width_cm >= 15.0 and data.Historical_Failure_Count >= 3 and data.Slope_Angle >= 65.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Terminal Aseismic Gravitational Shearing"
            prob_pct = 96.0
            boundary_exceeded = True

        # 36. Anthropogenic Surcharge on Liquefied Basal Strata
        elif data.Hill_Cutting_Severity >= 0.85 and data.Lithology_Weakness >= 0.85 and data.Soil_Saturation >= 0.90:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Anthropogenic Surcharge on Liquefied Basal Strata"
            prob_pct = 92.0
            boundary_exceeded = True

        # 37. Advanced Lithological Disintegration (Debris Flow)
        elif data.Lithology_Weakness >= 0.95 and data.Slope_Angle >= 55.0 and data.Rainfall_mm >= 60.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Advanced Lithological Disintegration and Debris Flow"
            prob_pct = 93.0
            boundary_exceeded = True

        # 38. Seismic Resonance Detachment of Fractured Rock Mass
        elif data.Surface_Crack_Width_cm >= 8.0 and data.Slope_Angle >= 70.0 and data.Earthquake_Activity >= 3.5:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Seismic Resonance Detachment of Fractured Rock Mass"
            prob_pct = 95.0
            boundary_exceeded = True

        # 39. Infrastructure-Induced Basal Plane Undermining
        elif data.Distance_to_Road_m <= 1.0 and data.Hill_Cutting_Severity >= 0.90 and data.Historical_Failure_Count >= 2:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Infrastructure-Induced Basal Plane Undermining"
            prob_pct = 91.0
            boundary_exceeded = True

        # 40. Complete Monsoonal Subgrade Liquefaction and Mass Wasting
        elif data.Rainfall_72h_mm >= 400.0 and data.Rainfall_mm >= 200.0 and data.Soil_Saturation >= 0.98:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Complete Monsoonal Subgrade Liquefaction and Mass Wasting"
            prob_pct = 99.9
            boundary_exceeded = True

        # 41. Aseismic Capillary Saturation and Base Yielding
        elif data.Rainfall_72h_mm == 0.0 and data.Rainfall_mm == 0.0 and data.Soil_Saturation >= 0.90 and data.Historical_Failure_Count >= 1:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Aseismic Capillary Saturation and Base Yielding"
            prob_pct = 85.0
            boundary_exceeded = True

        # 42. Traffic-Induced Acoustic Resonance on Fractured Overhang
        elif data.Slope_Angle >= 80.0 and data.Surface_Crack_Width_cm >= 10.0 and data.Distance_to_Road_m <= 1.0 and data.Hill_Cutting_Severity >= 0.80:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Traffic-Induced Acoustic Resonance on Fractured Overhang"
            prob_pct = 94.0
            boundary_exceeded = True

        # 43. Seismic Subsidence of Unconsolidated Fill
        elif data.Hill_Cutting_Severity >= 0.90 and data.Lithology_Weakness >= 0.90 and data.Earthquake_Activity >= 3.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Seismic Subsidence of Unconsolidated Fill"
            prob_pct = 90.0
            boundary_exceeded = True

        # 44. Expansive Soil Hydration and Strata Swell Collapse
        elif data.Soil_Saturation >= 0.95 and data.Lithology_Weakness >= 0.85 and data.Rainfall_mm <= 10.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Expansive Soil Hydration and Strata Swell Collapse"
            prob_pct = 88.0
            boundary_exceeded = True

        # 45. Pre-Seismic Creep Acceleration
        elif data.Surface_Crack_Width_cm >= 8.0 and data.Soil_Saturation >= 0.85 and data.Earthquake_Activity >= 2.5:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Pre-Seismic Creep Acceleration and Impending Shear"
            prob_pct = 92.0
            boundary_exceeded = True

        # 46. Hydro-Kinetic Shockwave Liquefaction
        elif data.Rainfall_mm >= 100.0 and data.Earthquake_Activity >= 4.5:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Hydro-Kinetic Shockwave Liquefaction"
            prob_pct = 97.0
            boundary_exceeded = True

        # 47. Undermined Structural Overhang Collapse
        elif data.Slope_Angle >= 85.0 and data.Hill_Cutting_Severity >= 0.95 and data.Historical_Failure_Count >= 1:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Undermined Structural Overhang Collapse"
            prob_pct = 96.0
            boundary_exceeded = True

        # 48. Catastrophic Highway Embankment Washout
        elif data.Rainfall_72h_mm >= 350.0 and data.Distance_to_Road_m <= 1.0 and data.Soil_Saturation >= 0.90:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Catastrophic Highway Embankment Washout"
            prob_pct = 98.0
            boundary_exceeded = True

        # 49. Gravitational Tearing of Degraded Escarpment
        elif data.Slope_Angle >= 70.0 and data.Lithology_Weakness >= 0.95 and data.Rainfall_72h_mm <= 10.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Gravitational Tearing of Degraded Escarpment"
            prob_pct = 89.0
            boundary_exceeded = True

        # 50. Absolute Structural Yield (Systematic Failure)
        elif data.Hill_Cutting_Severity == 1.0 and data.Lithology_Weakness >= 0.90 and data.Distance_to_Road_m <= 1.0 and data.Slope_Angle >= 60.0:
            alert_status = "DANGER 🔴 | CRITICAL EVENT: Absolute Structural Yield and Systematic Failure"
            prob_pct = 99.5
            boundary_exceeded = True

        # 51. Chronic Infrastructure Vulnerability (Warning Tier Fallback)
        elif data.Distance_to_Road_m <= 2.0 and data.Historical_Failure_Count >= 3:
            alert_status = "WARNING 🟠 | SYSTEM ALERT: Chronic Roadside Structural Degradation"
            prob_pct = 65.0
            boundary_exceeded = True

        # 4. Final Response Packaging
        return {
            "landslide_risk": 1 if "DANGER" in alert_status or boundary_exceeded else int(prediction),
            "probability_percentage": prob_pct,
            "status": alert_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))