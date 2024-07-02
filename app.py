from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import uuid
from api import infer

app = FastAPI()

@app.post("/generate_video/")
async def generate_video(image: UploadFile = File(...), audio: UploadFile = File(...)):
    image_path = f"/home/saad/Projects/SadTalker/sadtalker/tmp/{uuid.uuid4()}_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(image.file.read())

    audio_path = f"/home/saad/Projects/SadTalker/sadtalker/tmp/{uuid.uuid4()}_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(audio.file.read())

    # Assuming `sadtalker.process` is the function to generate video
    output_video_path = f"/home/saad/Projects/SadTalker/sadtalker/tmp/{uuid.uuid4()}_output.mp4"
    
    output_path = infer(audio_path, image_path)

    return FileResponse(output_path+'.mp4')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
