from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
from typing import List, Tuple
from contextlib import asynccontextmanager
import uvicorn
import pandas as pd
import io


class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    query: str
    annotation: List[Tuple[int, int, str]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    

    model_config = {
        "protected_namespaces": ()
    }


class CSVRowResponse(BaseModel):
    sample: str
    annotation: List[Tuple[int, int, str]]

ner_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ner_model
    try:
        from model.ner_model import initialize_model
        print("Загрузка модели")
        ner_model = initialize_model()
        if ner_model:
            print("Модель загружена")
        else:
            print(" Модель не загрузилась")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        ner_model = None
    
    yield 
    
    print("останавливается")

app = FastAPI(
    title="NER Model API",
    description="API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if ner_model is not None else "error",
        model_loaded=ner_model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        annotation = ner_model.predict_bio(request.query)
        return PredictionResponse(
            query=request.query,
            annotation=annotation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(request: List[PredictionRequest]):
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        results = []
        for item in request:
            annotation = ner_model.predict_bio(item.query)
            results.append({
                "query": item.query,
                "annotation": annotation
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
import ast
@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if ner_model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Проверяем что файл CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл в формате CSV")
        
        # Читаем CSV файл с обработкой ошибок
        contents = await file.read()
        df= pd.read_csv(io.BytesIO(contents), sep=';')  

        df["annotation"] = df["annotation"].apply(ast.literal_eval)
        

        
        if 'sample' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV 'sample'")
        
        # Обрабатываем датафрейм
        annotations = []
        for idx, row in df.iterrows():
            ann = ner_model.predict_bio(row['sample'])
            annotations.append(ann)
        
        df_result = df.copy()
        df_result['annotation'] = annotations
        
        output = io.BytesIO()
        df_result.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        filename = f"predictions_{file.filename}"
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки CSV файла: {str(e)}")

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "NER Model API", 
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict-batch": "/predict-batch (POST)", 
            "predict-csv": "/predict-csv (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)