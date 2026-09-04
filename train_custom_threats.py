import os
from ultralytics import YOLO

# ==============================================================================
# SCRIPT TO TRAIN YOLOv8 ON CUSTOM INTERNET DATASETS (WEAPONS / FIREARMS)
# ==============================================================================

def download_and_train():
    print("Initializing Custom Threat Training Pipeline...")
    
    # 1. Download Dataset from the Internet
    # We use Roboflow, the most common source for YOLOv8 datasets.
    # To run this, you need to 'pip install roboflow' and get a free API key from universe.roboflow.com.
    # Example Dataset: https://universe.roboflow.com/vcu-lziur/weapon-detection
    
    try:
        from roboflow import Roboflow
        # Replace 'YOUR_API_KEY' with your actual Roboflow API key
        rf = Roboflow(api_key="YOUR_API_KEY")
        
        print("Downloading Weapon Detection Dataset from Internet...")
        project = rf.workspace("vcu-lziur").project("weapon-detection-p09y5")
        version = project.version(1)
        dataset = version.download("yolov8")
        
        dataset_yaml = f"{dataset.location}/data.yaml"
        print(f"Dataset successfully downloaded to: {dataset_yaml}")
        
    except ImportError:
        print("ERROR: roboflow library not found. Install using: pip install roboflow")
        print("Falling back to assuming dataset is already downloaded locally as 'data.yaml'...")
        dataset_yaml = "data.yaml" # Fallback local file if Roboflow isn't used
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # 2. Initialize Pre-trained Model
    print("Loading base YOLOv8 model for transfer learning...")
    # We use yolov8n.pt as the foundation so it retains basic knowledge and trains quickly.
    model = YOLO("yolov8n.pt") 

    # 3. Train the Model on the Custom Dataset
    print("Starting Model Training...")
    print("This will process the internet dataset and teach the model what weapons look like.")
    
    # Training parameters:
    # - data: path to the YAML file describing the dataset classes and image paths.
    # - epochs: how many times it loops over the data (50-100 recommended for production).
    # - imgsz: image resolution (640 is standard).
    # - batch: how many images to process at once (16 is good for most hardware).
    
    try:
        results = model.train(
            data=dataset_yaml, 
            epochs=20, # Reduced for initial testing, use 100+ for high accuracy
            imgsz=640, 
            batch=16,
            name="border_weapons_model"
        )
        
        print("\n✅ Training Complete!")
        print("Your new weapon detection model is saved at: runs/detect/border_weapons_model/weights/best.pt")
        print("\nTo use this model in the main application, update border_surveillance.py:")
        print('Change: model = YOLO("yolov8n.pt")')
        print('To:     model = YOLO("runs/detect/border_weapons_model/weights/best.pt")')
        
    except Exception as e:
        print(f"\n❌ Training failed. Make sure your dataset is downloaded correctly. Error: {e}")

if __name__ == "__main__":
    # Ensure you are running this in a machine with sufficient memory/GPU if possible.
    download_and_train()
