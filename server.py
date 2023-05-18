from typing import Union, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import os

import boto3

from cv import emotionDetecte

app = FastAPI()

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

bucketname = 'parkinsense'

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def main():
    return {"Hello": "world"}

@app.post("/analyse")
async def analyseFile(files: List[UploadFile] = File(...) ):
    uploadPath = './uploades/'
    print(files)
    
    for file in files:
        contents = await file.read()

        with open(os.path.join(uploadPath, file.filename), "wb") as fp:
            fp.write(contents)
        print(file.filename + "is saved")

        result = emotionDetecte(file.filename)

        filepath = "./analyzed/"
        filename = file.filename.split(".png")[0] + "_analyzed.png"
        s3_client.upload_file(filepath + filename, bucketname, filename, ExtraArgs={'ContentType': "image/png"})

    return result