import os
import uuid
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form
from starlette.background import BackgroundTask
from moviepy.editor import VideoFileClip, AudioFileClip

from api import infer

app = FastAPI()

def cleanup_file(file_path: str):
    os.remove(file_path)
    print(f"File {file_path} has been deleted.")

@app.post("/generate_video/")
async def generate_video(
    image: UploadFile,
    audio: UploadFile,
    pose_style: int = File(0),
    batch_size: int = File(2),
    expression_scale: float = File(1.0),
    preprocess: str = File('crop'),
    still: bool = File(False),
    enhancer: str = File(None)
):

    image_path = f"tmp/{uuid.uuid4()}_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(image.file.read())

    audio_path = f"tmp/{uuid.uuid4()}_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(audio.file.read())

    print(image,audio,pose_style,batch_size,expression_scale,preprocess,still,enhancer)
    
    output_path = infer(
        driven_audio=audio_path,
        source_image=image_path,
        pose_style=pose_style,
        batch_size=batch_size,
        expression_scale=expression_scale,
        preprocess=preprocess,
        still=still,
        enhancer=enhancer
    )

    background_task = BackgroundTask(cleanup_file, output_path+".mp4")

    return FileResponse(output_path+".mp4", background=background_task)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
