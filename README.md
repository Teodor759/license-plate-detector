## <u>License Plate Detection & OCR System</u>

Don't expect much from this since I was kinda lazy, but I will improve it soon.

And I will also add the notebooks used to train and evaluate the models.

## 📌 Architecture & Training Details

The system splits the task into two core modules: **Object Detection** (localization) and **Optical Character Recognition** (text extraction).

| **Module**                | Dataset Source                                                                              | **Dataset Size** | **Status**        |
| ------------------------- | ------------------------------------------------------------------------------------------- | ---------------- | ----------------- |
| **Bounding Box Detector** | https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images | ~7K images       | Pretty decent     |
| **OCR System**            | https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset           | ~600 images      | Needs improvement |

## 🚀 Future Improvements

* Re-train the OCR model on an bigger dataset.

* Use `joblib`.


