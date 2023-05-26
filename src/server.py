from typing import Union, List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

import os
import base64

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
async def analyseFile(files: UploadFile = File(...) ):
    uploadPath = './uploads/'

    contents = await files.read()
    # print(contents)
    # print(files.filename)

    # with open(os.path.join(uploadPath, files.filename), "wb") as fp:
    #     fp.write(contents)
    # print(files.filename + " is saved")

    result = emotionDetecte(files.filename, contents)

    filepath = "./analyzed/"
    newFilename = files.filename.split(".png")[0] + "_analyzed.png"
    s3_client.upload_file(filepath + newFilename, bucketname, newFilename, ExtraArgs={'ContentType': "image/png"})

    return result
    # return True