## FER 
This folder is used to train and evaluate the performance of different models for different datasets for flipping error regularization and standard training without regularization. 

### Conda Environment
 You need to create a new conda environment using:
```
conda create -n fer python=3.8.18
```
 Then activate `fer` environment:
```
conda activate fer
```
And install the required libraries:
```
pip install -r requirements.txt
```

### Train
You can use the following command to train a model on the CIFAR-100 dataset using FER. Change the --model parameter to train on the other models and change the --dataset parameter to train on other datasets:
```
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 1 --dataset cifar100
```
Similarly, models can be trained using standard training without regularization by choosing parameter -i to be 240:
```
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
```
To train using Sari's EfficientNet model, use the following command:
```
python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 1 --dataset cifar100
```
To train using Juan's ResNeXt model, use the following command:
```
python train.py --model ResNeXt --kd_T 5 -s 7 --mu 1.0 -i 1 --dataset cifar100
```
To train using Kyle's regularization approach, use the following command with a specified -b beta value:
```
python ktrain.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 1 -b 1.0 --dataset cifar100
```
To train using Justin's approach, use the following command:
```
python jtrain.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 1 --dataset cifar100
```

### Test
After training, the model performance of the test data can be evaluated using the following command:
```
python test.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
```
