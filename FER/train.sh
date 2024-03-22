# Standard Training
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset stanford
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset tiny

python train.py --model GoogleNet --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
python train.py --model GoogleNet --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset stanford
python train.py --model GoogleNet --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset tiny

python train.py --model ResNeXt --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
python train.py --model ResNeXt --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset stanford
python train.py --model ResNeXt --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset tiny

python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 240 --dataset cifar100
python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 240 --dataset stanford
python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 240 --dataset tiny

python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 240 --dataset cifar100
python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 240 --dataset stanford
python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 240 --dataset tiny

python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 240 --dataset cifar100
python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 240 --dataset stanford
python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 240 --dataset tiny

python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset cifar100
python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset stanford
python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 240 --dataset tiny

# FER Training
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset cifar100
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset stanford
python train.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset tiny

python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 2 --dataset cifar100
python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 2 --dataset stanford
python train.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3 -i 2 --dataset tiny

python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 2 --dataset cifar100
python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 2 --dataset stanford
python train.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5 -i 2 --dataset tiny

python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 2 --dataset cifar100
python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 2 --dataset stanford
python train.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3 -i 2 --dataset tiny

python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset cifar100
python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset stanford
python train.py --model Efficient --kd_T 5 -s 7 --mu 1.0 -i 2 --dataset tiny
