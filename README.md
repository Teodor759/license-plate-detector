## <u>License Plate Detection & OCR System</u>

The system splits the task into two core parts: **Object Detection** (localization) and **Optical Character Recognition** (text extraction).

| **Module**                | Dataset Source                                                                              | **Dataset Size** |
| ------------------------- | ------------------------------------------------------------------------------------------- | ---------------- |
| **Bounding Box Detector** | https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images | ~7K images       |
| **OCR System**            | https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset, https://www.kaggle.com/datasets/luisasr2/brazilian-license-plate-ocr        | ~800 images      |


### Model Optimization & Size Reduction

The trained models (`best_detector_optimized.pth` and `best_ocr_optimized.pth`) were compressed using **dynamic quantization** – a PyTorch technique that converts floating‑point weights and activations to 8‑bit integers for linear and recurrent layers. This reduces the model size by approximately **75%** without significant loss of accuracy, and also speeds up inference on CPU.

**Important:** The quantized models are **CPU‑only** – they do not support GPU inference. The provided `app.py` runs everything on the CPU to ensure compatibility. If you need GPU support, you should skip quantization and keep the FP32 models.