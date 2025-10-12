# Real-Time Summons Detection

## Overview

This project aims to provide patrol officers with a more efficient way to enforce parking regulations by automatically detecting and issuing fines in real time. This approach minimises manual effort, improves accuracy, and enhances the overall efficiency of parking enforcement operations.

The system uses a **computer vision pipeline** built around a **Faster R-CNN** model, which performs **object detection** to locate number plates from vehicle images. The detected regions are then passed to **EasyOCR** for **optical character recognition (OCR)** to extract the plate text. Finally, the recognised text is filtered using **regular expressions (regex)** to ensure that only valid plate formats are accepted.  

Ideally, the plate numbers will be cross-referenced against a **summons database** to automatically flag vehicles with outstanding fines. However, currently the system uses a mock data array instead of a real database to simulate this lookup process.

### 🧠 Model Details
- **Model:** Faster R-CNN  
- **Mean Average Precision (mAP):** 0.7958 @ 0.5 IoU threshold  
- **Dataset:** 6,024 images of Malaysian number plates, including white-background plates (e.g., taxis, EVs).

### ⚠️ Limitations
- The model performs best on **flat, forward-facing images**. Images captured at an angle may still be detected but can cause **bounding box overshoot**, affecting downstream accuracy.  
- The **extracted bounding boxes** may include irrelevant text, reducing OCR precision. Additionally, **EasyOCR** may struggle with **fancy or stylised fonts**.  
- The **regex filtering** is strict. Even minor OCR errors (e.g., an extra character) can invalidate an otherwise correct result. Conversely, some incorrect outputs may still pass validation, leading to **false positives**.

## Usage

### 1. Configure the Summons Database
- Open `infer_and_ocr.py`.
- Navigate to **Line 19** (or wherever the `summons_db` array is defined).
- Add the **number plates** you want the system to flag in the array.

> 💡 Example:
> ```python
> summons_db = ["ABC1234", "WXY5678", "JKL9999"]
> ```

### 2. Add Images for Detection
- Place the images you want the system to analyse in the following directory: `data/unseen/`
- Each image should contain a **vehicle number plate** to be detected and processed.

### 3. Run the Inference Script
From the project root directory, execute the following command in your terminal:

```bash
python infer_and_ocr.py --folder ./data/unseen --model best_model.pth
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/thamjianhao/automatic-number-plate-recognition.git
cd automatic-number-plate-recognition
````

### 2. (Optional) Set Up a Virtual Environment

```bash
python -m venv venv
```

Activate it:

* **Windows**

  ```bash
  venv\Scripts\activate
  ```
  
* **macOS/Linux**
  
  ```bash
  source venv/bin/activate
  ```
  
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
