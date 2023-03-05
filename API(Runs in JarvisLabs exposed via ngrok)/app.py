from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse


from utils import summarize
from PortaSpeech.synthesize import get_audio

import shutil
import uvicorn

app = FastAPI()

@app.get("/summarize")
def index(url: str):
    return summarize(url)

@app.post("/summarize")
def index(myfile: UploadFile = File(...)):
    print(myfile.file)
    with open(myfile.filename, "wb+") as f:
        shutil.copyfileobj(myfile.file, f)
    return summarize(filepath = myfile.filename)

@app.get("/tts")
def index(text: str):
    get_audio(text)
    return FileResponse('./results.wav')

if __name__ == '__main__':
    uvicorn.run('app:app', host="0.0.0.0", port = 8000)