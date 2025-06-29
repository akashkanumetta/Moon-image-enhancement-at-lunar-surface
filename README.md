## Moon Surface Image Enhancement using FastAPI

This project focuses on enhancing low-light or shadowed images of the Moon’s surface, particularly from permanently shadowed regions (PSRs), using advanced image processing techniques. It is designed to improve visibility and detail in lunar imagery to support research and exploration missions.

Project Overview

The application is built with FastAPI and provides an interface to upload lunar surface images and receive enhanced versions using a combination of:

* Multi-Scale Retinex (MSR)
* Dark Channel Prior (DCP)
* Contrast Limited Adaptive Histogram Equalization (CLAHE)
* Unsharp Masking for edge enhancement
* Denoising using Non-Local Means

These methods work together to recover visual information from dark, low-contrast regions that are otherwise difficult to analyze.

Features

* FastAPI-based backend for image processing
* Real-time image upload and enhancement
* Automatic enhancement pipeline using fixed parameters
* Returns base64-encoded enhanced image for frontend display
* Supports JPG and PNG input formats

Dataset

The application is compatible with lunar surface images from scientific missions, particularly those affected by extreme shadows or low reflectance conditions. This includes data from:

* ISRO Chandrayaan-2 OHRC (Orbiter High Resolution Camera)
* NASA LRO (Lunar Reconnaissance Orbiter)
* Custom or pre-processed lunar datasets with visible terrain

The model assumes 3-channel images and applies preprocessing for better clarity before enhancement.

Technologies Used

* FastAPI for web API development
* OpenCV for image processing
* NumPy for numerical operations
* Base64 for image encoding
* Jinja2 for template rendering (optional HTML integration)

Use Cases

* Scientific research on lunar geology and topography
* Preprocessing for autonomous lunar rover navigation
* Enhancing PSR (Permanently Shadowed Region) images for exploration planning
* Visual analysis of shadowed craters and ridges

Future Improvements

* Integration with frontend for live preview and interaction
* Use of deep learning-based enhancement methods
* Automated crater detection from enhanced images
* Raspberry Pi-based deployment for edge computing scenarios
