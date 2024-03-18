## FER 
This folder is used to train and evaluate the performance of different models for different datasets for flipping error regularization and standard training without regularization. 

### Conda Environment
 `pytorch` conda enviroment was used which can be activated using the following command:
```
conda activate pytorch
```
All the required libraries to run the code is available in this environment.

### Train
You can use the following command to train a ResNet18 model on the CIFAR-100 dataset using FER
```
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset cifar100
```
The same model can be trained using standard training without regularization using:
```
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
```
### Test
After training, the model performance of the test data can be evaluated using the following command:
```
python test.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
```
