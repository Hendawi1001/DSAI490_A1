import os
import glob
import tensorflow as tf
import zipfile

def extract_data(zip_path, extract_path):
    if os.path.exists(zip_path):
        has_content = False
        if os.path.exists(extract_path) and os.listdir(extract_path):
            has_content = True
            
        if not has_content:
            os.makedirs(extract_path, exist_ok=True)
            print(f"Extracting {zip_path} to {extract_path} ")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        else:
            print(f"Content already exists in {extract_path}. Skipping extraction.")
    else:
        print(f"Zip file {zip_path} not found. Please ensure data is placed in {extract_path} directly.")

def get_regions(data_path):
    subdirs = [f.path for f in os.scandir(data_path) if f.is_dir()]
    regions = sorted([os.path.basename(d) for d in subdirs])
    return regions

def create_dataset(region_path, batch_size=64, img_size=(64, 64)):
    file_pattern = os.path.join(region_path, "*.jpeg")
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, img
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
