"""
The development of this pipeline is to predict the time needed for a person to colorize grayscale videos with graphics tools.
The core idea is to use a pre-trained CNN (YOLO11 in this case) to extract the mean values certain features from grayscale videos 
and use some pre-trained ML models to extract the time variable (deadline) as the mean of all the predictions.
The models used for this case all all that could be make a good fit to the training data.

The training data was extracted from hundreds of clips using the same function used to extract the features [person, object, speed, frames] from clips (process_clip).
As the clip selection and replacement of batches was done by hand, duplicates occured, they were seperated and augmented by slightly changing their values (noise - augmentation).

YOLO11 must be downloaded from the official ultralytics site 
https://docs.ultralytics.com/models/yolo11/
"""


import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import multiprocessing
from ultralytics import YOLO
import pickle

# Load the YOLO model
yolo_model = YOLO('yolo11x.pt')

# Definitions of paths for all folders needed.
clips_folder = './clips'
output_csv = './mean_values.xlsx'
models_folder = './working_models'
results_output = './mean_prediction_times.xlsx'


def process_clip(filename):
    file_path = os.path.join(clips_folder, filename)
    video_cap = cv2.VideoCapture(file_path)

    frame_count = 0
    people_count, object_count, flow_sum = 0, 0, 0
    prev_gray = None  # Initialize for optical flow

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        """
        OpenCV works often by turning the images to grayscale before processing.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # YOLO detection
        results = yolo_model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # Assuming YOLO outputs detections here

        # Count specific objects 
        people_count += sum(1 for detection in detections if detection[-1] == 0)  # Assuming class 0 is 'person'
        object_count += len(detections)

        """
        Optical flow is the displacement of pixel between successive images, namely in video files.
        When images move or the pictures are moving, optical flow increases. 
        The average optical flow is kept and recorded.
        """
        # Calculate optical flow for between successive frames
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_sum += magnitude.mean()  # Average flow magnitude

        prev_gray = gray

        frame_count += 1

    video_cap.release()
    """
    Simply calculate the frames of the clip.
    """
    if frame_count == 0:
        print(f"No frames processed for {filename}")
        return filename, {'clip': filename, 'person': 0, 'object': 0, 'speed': 0, 'frames': 0}

    # Calculate means
    mean_people = people_count / frame_count
    mean_objects = object_count / frame_count
    mean_flow = flow_sum / frame_count

    return filename, {
        'clip': filename,
        'person': mean_people,
        'object': mean_objects,
        'speed': mean_flow,
        'frames': frame_count
    }

def process_clip_wrapper(clip):
    """
    Wrapper for process_clip function for multiprocessing.
    """
    try:
        return process_clip(clip)
    except Exception as e:
        print(f"Error processing {clip}: {e}")
        return None

if __name__ == "__main__":
    # Get the list of clips
    clips = [clip for clip in os.listdir(clips_folder) if clip.endswith('.mp4')]

    # Use multiprocessing to process clips in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_clip_wrapper, clips)

    # Collect results into a DataFrame
    data = pd.DataFrame([result[1] for result in results if result is not None])

    # Save to Excel after processing all clips - safekeeping
    data.to_excel(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    # ----------------------------------------------------------------------------------------------------------------
    # Load all models
    models_dict = {}
    for model_file in os.listdir(models_folder):
        if model_file.endswith('.pkl'):  # Ensure only .pkl files are loaded
            model_path = os.path.join(models_folder, model_file)
            print('Loading model: ', model_file)
            with open(model_path, 'rb') as f:
                model_name = os.path.splitext(model_file)[0]
                models_dict[model_name] = pickle.load(f)

    print(f"Loaded {len(models_dict)} models: {list(models_dict.keys())}")

    # Initialize a list to store results
    predictions = []
    """
    We load the models and use each one to make a prediction for each clip.
    The mean value for the prediction of each clip is kept as the final result.
    """
    # Iterate over each clip in the data
    for _, row in data.iterrows():
        clip_name = row['clip']
        clip_features = row[['person', 'object', 'speed', 'frames']].values.reshape(1, -1)  # Reshape for model input

        clip_predictions = []  # Store predictions (times) for this clip across all models
    """
    One model was problematic during development and that is when the the try-except was added for error control.
    """
        for model_name, model in models_dict.items():
            try:
                prediction = model.predict(clip_features)
                clip_predictions.append(prediction[0] if hasattr(prediction, '__iter__') else prediction)
            except Exception as e:
                print(f"Error predicting with model {model_name} for clip {clip_name}: {e}")

        # Calculate mean prediction time for the clip across all models
        if clip_predictions:
            mean_prediction = sum(clip_predictions) / len(clip_predictions)
            predictions.append({'clip': clip_name, 'mean_prediction_time': round(mean_prediction, 2)})
        else:
            print(f"No predictions available for clip {clip_name}.")
            predictions.append({'clip': clip_name, 'mean_prediction_time': None})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(predictions)

    # Save the results to an Excel file
    results_df.to_excel(results_output, index=False)
    print(f"Saved mean prediction times to {results_output}")
