msg=final_v2.0_wo/centercrop

python train_RGB.py --message=$msg 
sleep 5
python train_RGB_Depth.py --message=$msg
sleep 5

#python test_RGB.py --message=$msg
sleep 5
#python test_RGB_Depth.py --message=$msg

