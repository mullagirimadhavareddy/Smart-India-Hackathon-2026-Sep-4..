import os
from ultralytics import YOLO

# ==============================================================================
# SCRIPT TO TRAIN YOLOv8 ON THE SNEHILSANYAL KAGGLE WEAPON DATASET
# Classes: Automatic Rifle, Bazooka, Grenade Launcher, Handgun, Knife, Shotgun, SMG, Sniper, Sword
# ==============================================================================

def download_and_train():
    print("Initializing Custom Threat Training Pipeline from Kaggle...")
    
    # 1. Download Dataset from Kaggle
    # Make sure you have your kaggle.json configured in ~/.kaggle/kaggle.json
    try:
        import kaggle
        print("Downloading dataset via Kaggle API...")
        kaggle.api.dataset_download_files('snehilsanyal/weapon-detection-test', path='kaggle_weapon_data', unzip=True)
        print("Download complete.")
    except Exception as e:
        print("Could not download via Kaggle API. Ensure you have the 'kaggle' library installed and your kaggle.json API key set up.")
        print("Error details:", e)
        print("Continuing assuming the data is already extracted to 'kaggle_weapon_data'...")

    # NOTE: The kaggle dataset provides YOLO format labels, but you may need to auto-generate a data.yaml.
    # A standard YOLO data.yaml looks like this:
    yaml_content = """
train: kaggle_weapon_data/train/images
val: kaggle_weapon_data/val/images

nc: 9
names: ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']
"""
    with open("weapon_data.yaml", "w") as f:
        f.write(yaml_content)

    # 2. Initialize Pre-trained Model
    print("\nLoading base YOLOv8 model for transfer learning...")
    model = YOLO("yolov8n.pt") 

    # 3. Train the Model on the Custom Dataset
    print("\nStarting Model Training...")
    try:
        results = model.train(
            data="weapon_data.yaml", 
            epochs=50,  # 50-100 recommended for production
            imgsz=640, 
            batch=16,
            name="border_weapons_model"
        )
        
        print("\n✅ Training Complete!")
        print("Your new 9-class weapon detection model is saved at: runs/detect/border_weapons_model/weights/best.pt")
        print("\nTo use this model in the main application, update border_surveillance.py:")
        print('Change: model = YOLO("yolov8n.pt")')
        print('To:     model = YOLO("runs/detect/border_weapons_model/weights/best.pt")')
        
    except Exception as e:
        print(f"\n❌ Training failed. Make sure your dataset is downloaded correctly. Error: {e}")

if __name__ == "__main__":
    download_and_train()
