# Plant Disease Detection App

In this project, I trained a CNN model for plant disease detection and deployed the model as a IOS app through CoreML by Apple.

**App Demo:**

![App Demo](reference/demo.gif)


Tutorial
-----

Dependencies:
- Keras
- CoreML


**1. Build and train a CNN model**
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

Open the starter app in Xcode, and drag PlantDisease.mlmodel from Finder into the project’s Project navigator. 
Select it to see the metadata you added:

<img src="/reference/add_model_to_App.png" alt="alt text" width="800" height="whatever">

Run predict through CoreML
```swift
func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    // load our CoreML Pokedex model
    guard let model = try? VNCoreMLModel(for: PlantDisease().model) else { return }
    
    // run an inference with CoreML
    let request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
        
        // grab the inference results
        guard let results = finishedRequest.results as? [VNClassificationObservation] else { return }
        
        // grab the highest confidence result
        guard let Observation = results.first else { return }
        
        // create the label text components
        let predclass = "\(Observation.identifier)"
        let predconfidence = String(format: "score: %.01f% %%", Observation.confidence * 100)
        
        // set the label text
        DispatchQueue.main.async(execute: {
            self.label.text = "\(predclass) \n \(predconfidence)"
        })
    }
    
    // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
    // Applications generating frames, compressing or decompressing video, or using Core Image
    // can all make use of Core Video pixel buffers
    guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    
    // execute the request
    try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
}

```

Check out more detail in [ViewController.swift](https://github.com/harrisonzzh/Plant_Disease_Detection/blob/master/App/PlantDisease/PlantDisease/ViewController.swift)

