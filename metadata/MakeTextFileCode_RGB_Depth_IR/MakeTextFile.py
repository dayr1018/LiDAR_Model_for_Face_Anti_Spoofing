import sys

def convert(num):
    if len(str(num)) == 1:
        return "00"+str(num)
    elif len(str(num)) == 2:
        return "0"+str(num)\

# Train Data
with open('train_list_All_protocol-1.txt', 'w') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
with open('train_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('train_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('train_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):
            file.write("Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_01_High/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open('train_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open('train_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(1,33):
        for jpgNum in range(1,31):            
            file.write("Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/depth/crop/" + convert(jpgNum) + ".jpg Training/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )


# Validation Data
with open('val_list_All_protocol-1.txt', 'w') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('val_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('val_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('val_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_01_High/real_01/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_01_High/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open('val_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open('val_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(33,41):
        for jpgNum in range(1,31):  
            file.write("Validation/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/depth/crop/" + convert(jpgNum) + ".jpg Validation/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )


# Test Data
with open('test_list_All_protocol-1.txt', 'w') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_01_High/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )
            
with open('test_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_02_Mid/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('test_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_03_Low/attack_07_3d_mask/ir/crop/" + convert(jpgNum) + ".jpg 0\n" )

with open('test_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_01_High/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )
            
with open('test_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):           
            file.write("Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_02_Mid/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )

with open('test_list_All_protocol-1.txt', 'a') as file:
    for fileNum in range(41,46):
        for jpgNum in range(1,31):  
            file.write("Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/color/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/depth/crop/" + convert(jpgNum) + ".jpg Test/" + str(fileNum) + "/KINECT/Light_03_Low/real_01/ir/crop/" + convert(jpgNum) + ".jpg 1\n" )



# Data Count 

file = open('train_list_All_protocol-1.txt', 'r')
print('train_list_All_protocol-1.txt')
print(file.read().count("\n")+1)
file.close()

file = open('val_list_All_protocol-1.txt', 'r')
print('val_list_All_protocol-1.txt')
print(file.read().count("\n")+1)
file.close()

file = open('test_list_All_protocol-1.txt', 'r')
print('test_list_All_protocol-1.txt')
print(file.read().count("\n")+1)
file.close()
