# Train file, Test file 두 가지가 필요함 
# Root 는 /mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS 임 
# 데이터는 총 1~12번까지 있으며, Test 데이터의 경우는 내가 임의로 추가해줘야 함 

Train_File = "train_data.txt"
Test_File = "test_data.txt"

Train_Start = 7
Train_End = 8 # means 5
Test_Start = 7
Test_End = 8 # means 10

facedata_start = 1
facedata_end = 21 # means 20

with open(Train_File, 'w') as file:
    for subNum in range(Train_Start,Train_End):
        for fileNum in range(facedata_start, facedata_end):
            file.write(f"LDFAS/{subNum}/bonafide/rgb_{fileNum}.jpg LDFAS/{subNum}/bonafide/depth_{fileNum}.jpg LDFAS/{subNum}/bonafide/pc_{fileNum}.ply 1\n")
        for fileNum in range(facedata_start, facedata_end):
            file.write(f"LDFAS/{subNum}/attack_mask/rgb_{fileNum}.jpg LDFAS/{subNum}/attack_mask/depth_{fileNum}.jpg LDFAS/{subNum}/attack_mask/pc_{fileNum}.ply 0\n")
        # for fileNum in range(facedata_start, facedata_end):
            #file.write(f"LDFAS/{subNum}/attack_paper/rgb_{fileNum}.jpg LDFAS/{subNum}/attack_paper/depth_{fileNum}.jpg LDFAS/{subNum}/attack_paper/pc_{fileNum}.ply 0\n")
    

with open(Test_File, 'w') as file:
    for subNum in range(Test_Start,Test_End):
        for fileNum in range(facedata_start, facedata_end):
            file.write(f"LDFAS/{subNum}/bonafide/rgb_{fileNum}.jpg LDFAS/{subNum}/bonafide/depth_{fileNum}.jpg LDFAS/{subNum}/bonafide/pc_{fileNum}.ply 1\n")
        for fileNum in range(facedata_start, facedata_end):
            file.write(f"LDFAS/{subNum}/attack_mask/rgb_{fileNum}.jpg LDFAS/{subNum}/attack_mask/depth_{fileNum}.jpg LDFAS/{subNum}/attack_mask/pc_{fileNum}.ply 0\n")
        # for fileNum in range(facedata_start, facedata_end):
            #file.write(f"LDFAS/{subNum}/attack_paper/rgb_{fileNum}.jpg LDFAS/{subNum}/attack_paper/depth_{fileNum}.jpg LDFAS/{subNum}/attack_paper/pc_{fileNum}.ply 0\n")



files = [Train_File, Test_File]

for filename in files:
    file = open(filename, 'r')
    print(filename)
    print(file.read().count("\n"))
    file.close()