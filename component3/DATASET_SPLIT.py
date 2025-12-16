
import os;
import random;
import shutil;

#directories
DATASET_DIR = 'dataset'

IMG_DIR = os.path.join(DATASET_DIR, 'images','all')
LBL_DIR = os.path.join(DATASET_DIR, 'labels','all')

TRAIN_split = 0.7 #70% for training
VAL_split = 0.15 #15% for validation
TEST_split = 0.15 #15% for testing

# Create directories for splits if they don't exist
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

# Get list of all image files
all_images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_images)

total = len(all_images)
train_end = int(total * TRAIN_split)
val_end = train_end + int(total * VAL_split)

# Split the dataset
train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

def move_files(file_list, split):
    for img in file_list:
        img_src = os.path.join(IMG_DIR, img)
        lbl_name = img.rsplit('.', 1)[0] + '.txt'
        lbl_src = os.path.join(LBL_DIR, lbl_name)

        img_dst = os.path.join(DATASET_DIR, 'images', split, img)
        lbl_dst = os.path.join(DATASET_DIR, 'labels', split, lbl_name)

        shutil.copy(img_src, img_dst)

        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
        else:
            print(f'Warning: Label file {lbl_src} not found for image {img_src}')


# Clear previous files in split directories
for split in ['train', 'val', 'test']:
    img_split_dir = os.path.join(DATASET_DIR, 'images', split)
    lbl_split_dir = os.path.join(DATASET_DIR, 'labels', split)
    
    for f in os.listdir(img_split_dir):
        os.remove(os.path.join(img_split_dir, f))
    
    for f in os.listdir(lbl_split_dir):
        os.remove(os.path.join(lbl_split_dir, f))


# Move files to respective directories
move_files(train_images, 'train')
move_files(val_images, 'val')
move_files(test_images, 'test')

print("\nDataset split completed!")