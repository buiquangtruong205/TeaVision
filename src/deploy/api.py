from fastapi import FastAPI


app = FastAPI(title="TeaVision API")


@app.get("/health")
def health():
    return {"status": "ok"}
