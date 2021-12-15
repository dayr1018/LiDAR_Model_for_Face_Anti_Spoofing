msg=cross-validation-fold-4
python train_RGB.py --message=$msg --skf=4
sleep 5
python train_RGB_Depth.py --message=$msg --skf=4
sleep 5

#python test_RGB.py --message=$msg
#sleep 5
#python test_RGB_Depth.py --message=$msg

