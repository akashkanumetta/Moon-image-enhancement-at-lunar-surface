from fastapi import FastAPI, Request, HTTPException, UploadFile, File, FormData
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import base64
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

# Database model
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

# Database connection
engine = create_engine(f'mysql+pymysql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}/{os.getenv("DB_NAME")}')
Base.metadata.create_all(engine)

# SMTP server configuration
smtp_server = os.getenv("SMTP_SERVER")
smtp_port = int(os.getenv("SMTP_PORT"))
smtp_username = os.getenv("SMTP_USERNAME")
smtp_password = os.getenv("SMTP_PASSWORD")

# Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to perform Single Scale Retinex
def single_scale_retinex(img, sigma):
    img_float = img.astype(np.float32) + 1.0
    gaussian_blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(gaussian_blur)
    return retinex

# Function to perform Multi-Scale Retinex (MSR)
def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    retinex /= len(sigmas)
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
    retinex = np.uint8(retinex)
    return retinex

# Function to apply Dark Channel Prior (DCP)
def dark_channel_prior(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

# Fixed parameters for MSR and DCP
def fixed_parameters():
    sigmas = [100, 1200, 100]
    dcp_size = 300
    return sigmas, dcp_size

# Function to apply Adaptive Histogram Equalization (CLAHE)
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img)
    clahe_channels = [clahe.apply(channel) for channel in channels]
    clahe_img = cv2.merge(clahe_channels)
    return clahe_img

# Function to apply Unsharp Masking
def apply_unsharp_mask(img, blur_size=(9, 9), amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(img, blur_size, 0)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def enhanced_msr_dcp(img):
    sigmas, dcp_size = fixed_parameters()

    img_msr = np.zeros_like(img)
    for i in range(3):
        img_msr[:, :, i] = multi_scale_retinex(img[:, :, i], sigmas)

    dark_channel = dark_channel_prior(img_msr, size=dcp_size)
    transmission = 1 - dark_channel / 255.0
    transmission = cv2.blur(transmission, (dcp_size, dcp_size))

    img_msr_dcp = np.zeros_like(img_msr, dtype=np.float32)
    for i in range(3):
        img_msr_dcp[:, :, i] = (img_msr[:, :, i] - (1 - transmission) * np.min(img_msr)) / (transmission + 1e-4)

    img_msr_dcp = np.clip(img_msr_dcp, 0, 255).astype(np.uint8)

    img_clahe = apply_clahe(img_msr_dcp)

    img_sharpened = apply_unsharp_mask(img_clahe, blur_size=(9, 9), amount=1.5, threshold=10)

    return img_sharpened

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def retinex_color_restoration(img, sigmas):
    img_msr = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        img_msr += multi_scale_retinex(img, [sigma])
    img_msr /= len(sigmas)
    img_msr = cv2.normalize(img_msr, None, 0, 255, cv2.NORM_MINMAX)
    img_msr = np.uint8(img_msr)
    return img_msr

# FastAPI endpoint for user registration
@app.post("/register")
async def register_user(user: User):
    db = Session(engine)
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    user.password = generate_password_hash(user.password, method="sha256")
    db.add(user)
    db.commit()
    db.refresh(user)

# FastAPI endpoint for user login
@app.post("/login")
async def login_user(username: str, password: str):
    db = Session(engine)
    user = db.query(User).filter(User.username == username).first()

    if user and check_password_hash(user.password, password):
        return {"detail": "Logged in successfully"}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

# FastAPI endpoint for image upload and enhancement
@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Image enhancement
        enhanced_img = enhanced_msr_dcp(img)

        _, buffer = cv2.imencode('.jpg', enhanced_img)
        base64_image = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

        return {"image_data": base64_image}
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image:\n{str(e)}")

# FastAPI endpoint for user logout
@app.get("/logout")
async def logout_user():
    return {"detail": "Logged out successfully"}

# FastAPI endpoint for checking if the user is logged in
@app.get("/check_login")
async def check_login():
    return {"logged_in": False}  # Replace with actual login status check

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)