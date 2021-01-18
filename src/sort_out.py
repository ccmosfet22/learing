import random
import os
import shutil


def batch_rename(path):
    count = 0
    for fname in os.listdir(path):
        new_fname = str(count)
        print(os.path.join(path, fname))
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))
        count = count + 1


def create_folder(path):
    normal = []
    abnormal = []
    for main_folder in os.listdir(path):

        # print(main_folder)
        if main_folder == "normal":

            normal_path = "./train_data/normal"

            if not os.path.exists(normal_path):  # 如果資料夾不存在就建立
                os.makedirs(normal_path)

            for root, dirs, files in os.walk(path+"/"+main_folder):

                for name in files:
                    # print(root, name)
                    normal.append(root+"/"+name)

        elif main_folder == "abnormal":
            # print(123)
            abnormal_path = "./train_data/abnormal"

            if not os.path.isdir(abnormal_path):  # 如果資料夾不存在就建立
                os.makedirs(abnormal_path)

            for root, dirs, files in os.walk(path+"/"+main_folder):

                for name in files:

                    abnormal.append(root+"/"+name)

    return normal, abnormal, normal_path, abnormal_path


def move_file(origin_path, new_path):

    print(new_path)
    for a in origin_path:

        shutil.copy(a, new_path)


def copyFile(fileDir, desDir):

    # 1
    pathDir = os.listdir(fileDir)
    print(len(pathDir))
    # # 2
    sample = random.sample(pathDir, 10564)
    # # print sample

    if not os.path.isdir(desDir):  # 如果資料夾不存在就建立
        os.makedirs(desDir)
    # print(123)
    # # 3
    # for name in sample:
    #     shutil.move(fileDir+"/"+name, desDir+"/"+name)


if __name__ == "__main__":

    path = "C:\\Users\\YF\\Desktop\\Chainwin\\depth_image_analysis"
    save_path = "./"

    normal, abnormal, normal_path, abnormal_path = create_folder(path)

    # print(normal)
    # move_file(normal, normal_path)
    # move_file(abnormal, abnormal_path)

    desDir = "./train_data/normal"
    copyFile(abnormal_path, desDir)
