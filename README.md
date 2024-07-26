All code is inside source code directory

Create environment

conda create --name gauge_reading python==3.8
conda activate gauge_reading

pip install -r requirements.txt

For inference

python read_gauge_meter.py --image_path {your root director}/Submit/dataset/realdataset/test/images --model_path {your root director}/Submit/models/finetunned_realdata_11.pth --val_file {your root director}/Submit/validation.csv

image_path=>location of your image
model_path=> Trained model image_path
val_file=>.csv with groundtruth


For Training:
You can directly run train.ipynb
provide required fields: 
KEYPOINTS_FOLDER_TRAIN = '{your root director}/Submit/dataset/realdataset/train'
KEYPOINTS_FOLDER_TRAIN = '{your root director}/Submit/dataset/realdataset/train'
KEYPOINTS_FOLDER_TEST  = '{your root director}/Submit/dataset/realdataset/test'
WEIGHT_PATH            = '{your root director}/Submit/models/synthetic_data_model_8.pth'