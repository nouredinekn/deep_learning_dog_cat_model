# Dog vs Cat Classifier

This project provides a pre-trained model (`dog_cat.h5`) to classify images as either a cat or a dog.

---

## Features
- Pre-trained model ready for use.
- Simple pipeline to predict:
  - **0**: Dog
  - **1**: Cat

---

## Requirements
Install the following Python libraries:
- TensorFlow
- Keras
- NumPy
- Pillow

---

## Setup and Usage

### 1. Place Your Image
Place your image (e.g., `test.jpg`) in the project directory or specify its path.

### 2. Use the Model
Run the following script to make a prediction:

```python
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("dog_cat.h5")

# Load and preprocess the image
image_path = "test.jpg"  # Replace with your image path
img = Image.open(image_path).resize((24, 24)).convert("L")
img_array = np.array(img) / 255.0  # Normalize pixel values

# Make a prediction
prediction = np.argmax(model.predict(np.array([img_array])))
label = "Dog" if prediction == 0 else "Cat"
print(f"Prediction: {label}")
```

### 3. Output
The script will output:
- **`Dog`** if the prediction is `0`.
- **`Cat`** if the prediction is `1`.

---

## File Structure
```
dog-cat-classifier/
│
├── dog_cat.h5
├── test.jpg
└── README.md
```

---

## Author
Developed by **Nouredine_kn**  
- **Instagram**: [@nouredine_kn](https://instagram.com/nouredine_kn)  
- **Telegram**: [@nouredine_kn](https://t.me/nouredine_kn)

---

## Notes
- The model expects grayscale images resized to 24x24 pixels. Images are automatically preprocessed in the script.
- The classifier was trained for demonstration purposes and may not generalize to all datasets.
