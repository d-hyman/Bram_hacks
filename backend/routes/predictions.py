from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.data_manager import DataManager
from models.ml_models import MLModels

router = APIRouter()
data_manager = DataManager()
ml_models = MLModels()

class PredictionRequest(BaseModel):
    forest_cover_2000: float
    total_area: float
    model_type: str = "random_forest"

@router.on_event("startup")
async def train_models_on_startup():
    """Train models when router is loaded"""
    try:
        df = data_manager.get_data()
        ml_models.train(df)
    except:
        pass  # Will be trained on first request if this fails

@router.get("/models/performance")
async def get_model_performance():
    """Get ML model performance metrics"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        if not ml_models.is_trained():
            df = data_manager.get_data()
            ml_models.train(df)
        
        metrics = ml_models.get_metrics()
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict_forest_cover(request: PredictionRequest):
    """Predict 2010 forest cover based on 2000 data"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        if not ml_models.is_trained():
            df = data_manager.get_data()
            ml_models.train(df)
        
        prediction = ml_models.predict(
            request.forest_cover_2000,
            request.total_area,
            request.model_type
        )
        
        return {
            "success": True,
            "data": {
                "model_used": request.model_type,
                "input": {
                    "forest_cover_2000_percent": request.forest_cover_2000,
                    "total_area_km2": request.total_area
                },
                "predicted_forest_cover_2010_percent": prediction
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
