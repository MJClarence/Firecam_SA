import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image

model_dirpath = '..../Azure_Iteration_5_SM/'
pre_image_dir = '..../Testing_Pre/'
post_image_dir = '..../Testing_Post/'

PROB_THRESHOLD = 0.1  # Minimum probability to show results.

def load_model(model_dirpath):
    model = tf.saved_model.load(str(model_dirpath))
    serve = model.signatures['serving_default']
    input_shape = serve.inputs[0].shape[1:3]
    return serve, input_shape

def predict(model_serve, input_shape, image_filepath):
    image = Image.open(image_filepath).resize(input_shape)
    input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]

    input_tensor = tf.convert_to_tensor(input_array)
    outputs = model_serve(input_tensor)
    return {k: v[np.newaxis, ...] for k, v in outputs.items()}


model_serve, input_shape = load_model(model_dirpath)
all_files = os.listdir(pre_image_dir)

# Iterate through the files and delete non-JPG files
for file_name in all_files:
    if not file_name.endswith(".jpg"):
        file_path = os.path.join(pre_image_dir, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        
for filename in os.listdir(pre_image_dir):
    if filename.endswith('.DS_Store'):
        print("not an image")
    else: 
        image_path = os.path.join(pre_image_dir, filename)
    
        original_image = os.path.join(pre_image_dir, filename)
           
        outputs = predict(model_serve, input_shape, image_path)
        
        # Check if the required keys are present in the outputs
        assert set(outputs.keys()) == set(['detected_boxes', 'detected_classes', 'detected_scores'])
        
        # Initialize variables
        prob, x1, y1, x2, y2 = None, None, None, None, None
        
        # Iterate through the results and set variables based on the probability threshold
        for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], \
                                        outputs['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                prob = score
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if prob is None:
                    print("No fire detected")  
                else:
                    image = Image.open(image_path)
                    
                    # Get width and height of the image
                    W, H = image.size
                    
                    con = float(prob)
                    x0 = int(float(x1) * W)
                    x1 = int(float(x2) * W)
                    y0 = int(float(y1) * H)
                    y1 = int(float(y2) * H)
                    
                    img = cv2.imread(image_path)
                    
                    confidence = "Smoke: " + str(round(100 * con, 1)) + "%"
                    con = round(100 * con, 1)
                    # x0, y0, x1, y1 = map(int, bounding_box[:4])
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(img, confidence, (x0, y0 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    #Save inference image
                    output_path = os.path.join(post_image_dir, filename)
                    cv2.imwrite(output_path, img) 

      
