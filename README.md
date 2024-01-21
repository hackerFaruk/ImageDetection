# ImageDetection


Bu Repo image detection konularında farklı modelleri kullanma ve eğitme becerilerinin kazanılması için başlatılan bir kaynaktır. Farklı görüntü tanıma teknolojisi konseptlerini keşfetmeyi, sade ve anlaşılır süreçlerle ifade etmeyi hedefler.


## Modeli Kullanmak İçin

```
# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

# Initialization
base_options = core.BaseOptions(file_name=model_path)
detection_options = processor.DetectionOptions(max_results=2)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)

```






## Kullanılan Kaynklar 
- TensorFlow Lİte tutorial notebook
```
https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb#scrollTo=35BJmtVpAP_n
```


- Evan Juras, EJ Technology Consultants TensorFlow Lite Object Detection API in Colab
```
https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb
```

- COCO Evaluate
```
https://cocodataset.org/#detection-eval
```

- TensorFlow Docs Use model in python
```
https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector
```
