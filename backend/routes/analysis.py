from fastapi import APIRouter, HTTPException
from models.data_manager import DataManager
from models.statistics import StatisticsCalculator

router = APIRouter()
data_manager = DataManager()
stats_calc = StatisticsCalculator()

@router.get("/correlation")
async def get_correlation_matrix():
    """Get correlation matrix of forest variables"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        df = data_manager.get_data()
        corr_data = stats_calc.get_correlation_matrix(df)
        return {"success": True, "data": corr_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution")
async def get_distribution_statistics():
    """Get distribution statistics for forest cover change"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        df = data_manager.get_data()
        dist_stats = stats_calc.get_distribution_statistics(df)
        return {"success": True, "data": dist_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistical-tests")
async def get_statistical_tests():
    """Perform statistical tests on the data"""
    try:
        if not data_manager.is_initialized():
            data_manager.initialize()
        
        df = data_manager.get_data()
        test_results = stats_calc.perform_statistical_tests(df)
        return {"success": True, "data": test_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))