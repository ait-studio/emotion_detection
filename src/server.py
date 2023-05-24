from typing import Union, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

import os

import boto3

from cv import emotionDetecte

app = FastAPI()

load_dotenv()

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SKEY = os.getenv("AWS_SECRET_ACCESS_KEY")
print("GOT AWS ACCESS INFOS : ")
print("AWS_ACCESS_KEY_ID : ", AWS_KEY)
print("AWS_SECRET_ACCESS_KEY : ", AWS_SKEY)

session = boto3.Session(
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SKEY
)

s3 = session.resource('s3')
s3_client = session.client('s3')

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
    uploadPath = './uploads/'
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