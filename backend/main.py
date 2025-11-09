from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.data_manager import DataManager
from routes import statistics, predictions, analysis

app = FastAPI(title="Forest Cover Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data manager
data_manager = DataManager()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    try:
        data_manager.initialize()
        if data_manager.is_initialized():
            print("✓ Data loaded successfully")
        else:
            print("✗ Data initialization failed")
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        import traceback
        traceback.print_exc()

# Include routers
app.include_router(statistics.router, prefix="/api/statistics", tags=["statistics"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

@app.get("/")
async def root():
    return {
        "message": "Forest Cover Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "statistics": "/api/statistics",
            "predictions": "/api/predictions",
            "analysis": "/api/analysis"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": data_manager.is_initialized()
    }

