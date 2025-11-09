from fastapi import APIRouter, HTTPException
from models.data_manager import DataManager
from models.statistics import StatisticsCalculator

router = APIRouter()
# Use the singleton instance - same as main.py
data_manager = DataManager()
stats_calc = StatisticsCalculator()

@router.get("/overview")
async def get_overview_statistics():
    """Get overall forest cover statistics"""
    try:
        # Ensure data is initialized (fallback if startup didn't run)
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        df = data_manager.get_data()
        stats = stats_calc.get_overview_statistics(df)
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/country/{country_name}")
async def get_country_statistics(country_name: str):
    """Get statistics for a specific country"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        row = data_manager.get_country_data(country_name)
        
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Country '{country_name}' not found"
            )
        
        return {
            "success": True,
            "data": {
                "country": row['country'],
                "total_area_km2": float(row['area']),
                "forest_cover_2000_percent": float(row['two_thousand_percent']),
                "forest_cover_2000_km2": float(row['two_thousand_area']),
                "forest_cover_2010_percent": float(row['two_thousand_ten_percent']),
                "forest_cover_2010_km2": float(row['two_thousand_ten_area']),
                "change_percent": float(row['delta_percent']),
                "change_km2": float(row['delta_area']),
                "change_type": (
                    "reforestation" if row['delta_percent'] > 0
                    else "deforestation" if row['delta_percent'] < 0
                    else "no_change"
                )
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-deforestation")
async def get_top_deforestation(limit: int = 10):
    """Get countries with highest deforestation"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        deforestation = data_manager.get_top_deforestation(limit)
        
        return {
            "success": True,
            "data": [
                {
                    "country": row['country'],
                    "forest_loss_percent": float(abs(row['delta_percent'])),  # Convert to positive for loss
                    "forest_loss_km2": float(abs(row['delta_area'])),  # Convert to positive for loss
                    "forest_cover_2000": float(row['two_thousand_percent']),
                    "forest_cover_2010": float(row['two_thousand_ten_percent'])
                }
                for _, row in deforestation.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-reforestation")
async def get_top_reforestation(limit: int = 10):
    """Get countries with highest reforestation"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        reforestation = data_manager.get_top_reforestation(limit)
        
        return {
            "success": True,
            "data": [
                {
                    "country": row['country'],
                    "forest_gain_percent": float(row['delta_percent']),  # Already positive for gain
                    "forest_gain_km2": float(row['delta_area']),  # Already positive for gain
                    "forest_cover_2000": float(row['two_thousand_percent']),
                    "forest_cover_2010": float(row['two_thousand_ten_percent'])
                }
                for _, row in reforestation.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/countries")
async def get_all_countries():
    """Get list of all countries with basic stats"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        df = data_manager.get_data()
        
        return {
            "success": True,
            "data": [
                {
                    "country": row['country'],
                    "forest_cover_2000": float(row['two_thousand_percent']),
                    "forest_cover_2010": float(row['two_thousand_ten_percent']),
                    "change_percent": float(row['delta_percent']),
                    "change_type": (
                        "gain" if row['delta_percent'] > 0
                        else "loss" if row['delta_percent'] < 0
                        else "stable"
                    )
                }
                for _, row in df.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

