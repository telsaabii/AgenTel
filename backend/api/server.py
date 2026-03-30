from fastapi import FastAPI
from backend.api.context import router as context_router

app = FastAPI(title="AgenTel API", version="0.1.0")

app.include_router(context_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
