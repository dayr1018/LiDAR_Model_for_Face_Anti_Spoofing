python test.py --model rgb --attacktype rpm --epochs 500 --cuda 1 --message 0825_total_rgb &
python test.py --model rgbdp_v3 --attacktype rpm --epochs 500 --cuda 0 --message 0828_total_rgbdp_v3 &
python test.py --model rgbdp_v1 --attacktype rpm --epochs 500 --cuda 0 --message 0828_total_rgbdp_v1

python test.py --model rgb --attacktype rpm --epochs 500 --cuda 1 --message 0909_total_rgb &
python test.py --model rgbdp_v3 --attacktype rpm --epochs 500 --cuda 0 --message 0909_total_rgbdp_v3 &
python test.py --model rgbdp_v1 --attacktype rpm --epochs 500 --cuda 0 --message 0909_total_rgbdp_v1
