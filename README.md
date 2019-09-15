# Plant Disease Detection App

In this project, I trained a CNN model for plant disease detection and deployed the model as a IOS app through CoreML by Apple.

**App Demo:**

![App Demo](reference/demo.gif)


Tutorial
-----

Dependencies:
- Keras
- CoreML


**1. Build and train a CNN model: **
Checkout more detail in the [Notebook](https://github.com/harrisonzzh/Plant_Disease_Detection/blob/master/plant-disease-detection-using-keras.ipynb).

**2. Convert to a IOS model**

```python
from keras.models import load_model
import coremltools
import re

# rename the label App display

clean_label = []
for l in label_binarizer.classes_.tolist():
    name = re.split("[_]+", l.lower())
    crop = name[0]
    disease = " ".join(name[1:])
    clean_name = "{}: {}".format(crop, disease)
    clean_label.append(clean_name)
    print(clean_name)

# convert model
print("[INFO] converting model")
class_labels = label_binarizer.classes_.tolist()
coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names="image",
                                                    image_input_names="image",
                                                    image_scale=1/255.0,
                                                    class_labels=class_labels,
                                                    is_bgr=True)
# save
output = 'PlantDisease.mlmodel'
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)

```

**3. Integrate to your APP**



