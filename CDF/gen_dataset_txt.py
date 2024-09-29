import glob
import os
import cv2
import random
from skimage.exposure import rescale_intensity
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import trange

txt_name_CD = 'image_CD_labels.txt'
txt_name = 'images.txt'
txt_name2 = 'image_class_labels.txt'
txt_name3 = 'train_test_split.txt'
root_path = '/dataset_txt'
if not os.path.exists(root_path):
    os.makedirs(root_path)

txt_path_CD = os.path.join(root_path, txt_name_CD)
txt_path = os.path.join(root_path, txt_name)
txt_path2 = os.path.join(root_path, txt_name2)
txt_path3 = os.path.join(root_path, txt_name3)

txt_path_CDCD = open(txt_path_CD, 'w')
txt_file1 = open(txt_path, 'w')
txt_file22 = open(txt_path2, 'w')
txt_file33 = open(txt_path3, 'w')

txt_fileCD = open(txt_path_CD, 'a')
txt_file = open(txt_path, 'a')
txt_file2 = open(txt_path2, 'a')
txt_file3 = open(txt_path3, 'a')

def deconcolor(imageIn, removeBlack=True):
    global patch_assess
    patch_assess=None
    # input image
    numb_channels = 3
    patch_image = Image.open(imageIn)
    patch_image = patch_image.convert("RGB")
    patch_image = np.array(patch_image)
    patch_shape = patch_image.shape
    null = np.zeros_like(patch_image[:, :, 0])

    # IHC_hed = rgb2hed(patch_image)
    # IHC_h = hed2rgb(np.stack((IHC_hed[:, :, 0], null, null), axis=-1))
    # IHC_d = hed2rgb(np.stack((null, null, IHC_hed[:, :, 2]), axis=-1))

    Imax = np.max(patch_image)
    # define the color values
    #    R      G     B
    M = np.array([[0.18, 0.20, 0.08],  # Hematoxylin
                  [0.10, 0.21, 0.29]])  # DAB

    numb_stains = M.shape[0]

    for i in range(M.shape[0]):
        M[i, :] = M[i, :] / np.linalg.norm(M[i, :])

    imageOD = -np.log10((patch_image + 1.) / Imax)

    imageOD = np.reshape(imageOD, (-1, numb_channels))

    imageDecon = np.dot(imageOD, np.linalg.pinv(M))

    # reverse deconvolution
    imageDecon = Imax * np.exp(-imageDecon)
    imageDecon = np.clip(imageDecon, 0, 255)

    imageDecon = np.reshape(imageDecon, (patch_shape[0], patch_shape[1], numb_stains))

    DAB_info = rescale_intensity(imageDecon[:, :, 1], out_range=(0, 1)
                          , in_range=(0, np.percentile(imageDecon[:, :, 1], 100)))

    DAB_info = DAB_info * 255
    DAB_info = DAB_info.astype("uint8")

    return DAB_info

def hist_info(imageIn):
    patch_image = cv2.imread(imageIn)
    hist = cv2.calcHist([patch_image], [0], None, [256], [0, 256])

    patch_image_pixel = []
    [rows, cols] = patch_image.shape
    for i in range(rows):
        for j in range(cols):
            patch_image_pixel.append(patch_image[i, j])
    # fig = plt.figure(figsize=(18, 10))
    # print("patch_image_pixel", patch_image_pixel)
    DAB_range = [0, 256]  # x轴范围
    bins = np.arange(DAB_range[0], DAB_range[1], 1)  # 其中的1是每个bin的宽度。这个值取小，可以提升画图的精度
    # print("bins", bins)

    counts, _, _ = plt.hist(patch_image_pixel, bins, color='blue', alpha=0.5)  # alpha设置透明度，0为完全透明
    # print("counts", counts)
    counts = list(counts)
    return counts


def write_txt(Negative_image_path, Negative_mode, Positive_image_path, Positive_mode, CD_img_save_path, seed_num, data_opt):
    Negative_path = glob.glob(Negative_image_path + '/*')
    Positive_path = glob.glob(Positive_image_path + '/*')
    CD_path = glob.glob(CD_img_save_path + '/*')
    print("CD_path Len:", len(CD_path))
    random.seed(seed_num)
    index = 1
    Negative_index = 0
    Positive_index = 0

    Negative_train_flag = int(len(Negative_path) * 0.6)
    Negative_val_flag = len(Negative_path) - Negative_train_flag
    Positive_train_flag = int(len(Positive_path) * 0.6)
    Positive_val_flag = len(Positive_path) - Positive_train_flag

    Negative_one_list = [1 for index in range(Negative_train_flag)]
    Negative_zero_list = [0 for index in range(Negative_val_flag)]
    Negative_final_list = Negative_zero_list + Negative_one_list
    random.shuffle(Negative_final_list)
    Positive_one_list = [1 for index in range(Positive_train_flag)]
    Positive_zero_list = [0 for index in range(Positive_val_flag)]
    Positive_final_list = Positive_zero_list + Positive_one_list
    random.shuffle(Positive_final_list)

    print(Negative_mode, " train >", Negative_train_flag, " val >", Negative_val_flag)
    print(Positive_mode, " train >", Positive_train_flag, " val >", Positive_val_flag)

    # print("Negative_one_list >", len(Negative_one_list), " Negative_zero_list >", len(Negative_zero_list))
    # print("Positive_one_list >", len(Positive_one_list), " Negative_zero_list >", len(Negative_zero_list))
    kernal_size = 3
    print("Gen Negative Dataset List")
    for num in trange(len(Negative_path)):
        (patch_path, patch_extname) = os.path.split(Negative_path[num])
        (patch_name, extension) = os.path.splitext(patch_extname)

        index = str(index)
        number = '1'
        img_path_name = Negative_mode + '/' + patch_extname
        if data_opt == 'CD':
            CD_img = deconcolor(Negative_path[num])
            # CD_img = cv2.medianBlur(CD_img, ksize=kernal_size)
            DAB_patch_path = CD_img_save_path + '/' + '{}.png'.format(index)
            cv2.imwrite(DAB_patch_path, CD_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损
        # elif data_opt == 'hist':
        #     CD_info = deconcolor(Negative_path[num])
        #     CD_info = ",".join(str(i) for i in CD_info)
        #     txt_fileCD.writelines([index, ' ', CD_info, '\n'])
        else:
            CD_src_path = CD_img_save_path + '/' + '{}_Neg.png'.format(patch_name)
            CD_target_path = CD_img_save_path + '/' + '{}.png'.format(index)
            os.rename(CD_src_path, CD_target_path)
        txt_file.writelines([index, ' ', img_path_name, '\n'])
        txt_file2.writelines([index, ' ', number, '\n'])
        txt_file3.writelines([index, ' ', str(Negative_final_list[Negative_index]), '\n'])
        index = int(index)
        Negative_index += 1
        index += 1

    print("Gen Positive Dataset List")
    for num in trange(len(Positive_path)):
        (patch_path, patch_extname) = os.path.split(Positive_path[num])
        (patch_name, extension) = os.path.splitext(patch_extname)
        index = str(index)
        number = '2'
        img_path_name = Positive_mode + '/' + patch_extname
        if data_opt == 'CD':
            CD_img = deconcolor(Positive_path[num])
            # CD_img = cv2.medianBlur(CD_img, ksize=kernal_size)
            DAB_patch_path = CD_img_save_path + '/' + '{}.png'.format(index)
            cv2.imwrite(DAB_patch_path, CD_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损
        # elif data_opt == 'hist':
        #     CD_info = deconcolor(Positive_path[num])
        #     CD_info = ",".join(str(i) for i in CD_info)
        #     txt_fileCD.writelines([index, ' ', CD_info, '\n'])
        else:
            CD_src_path = CD_img_save_path + '/' + '{}_Pos.png'.format(patch_name)
            CD_target_path = CD_img_save_path + '/' + '{}.png'.format(index)
            os.rename(CD_src_path, CD_target_path)
        txt_file.writelines([index, ' ', img_path_name, '\n'])
        txt_file2.writelines([index, ' ', number, '\n'])
        txt_file3.writelines([index, ' ', str(Positive_final_list[Positive_index]), '\n'])
        index = int(index)
        Positive_index += 1
        index += 1


def main(images_root, GHA_dir, seed_num):
    dataset_name = 'DeepLIIF'
    Negative_image_path = '/data/{}/{}/Negative'.format(dataset_name, images_root)
    Positive_image_path = '/data/{}/{}/Positive'.format(dataset_name, images_root)

    CD_img_save_path = '/data/{}/{}'.format(dataset_name, GHA_dir)
    if not os.path.exists(CD_img_save_path):
        os.makedirs(CD_img_save_path)
    CD_img_path = glob.glob(CD_img_save_path + "/*")
    print("CD img Len >", len(CD_img_path))
    for patch in CD_img_path:
        os.remove(os.path.join(patch))
    print("Delete CD img Finish !")

    data_opt = 'CD' # mask |  CD
    if data_opt == 'CD':
        print("data_opt:", data_opt)
    else:
        print("data_opt:", data_opt)

        Negative_all_image_path = glob.glob(Negative_image_path + "/*")
        print("Negative img Len >", len(Negative_all_image_path))
        for patch in Negative_all_image_path:
            os.remove(os.path.join(patch))
        print("Delete Negative img Finish !")
        Positive_all_image_path = glob.glob(Positive_image_path + "/*")
        print("Positive img Len >", len(Positive_all_image_path))
        for patch in Positive_all_image_path:
            os.remove(os.path.join(patch))
        print("Delete Positive img Finish !")

        Input_dataset_path = '/media/L/NFY/Final_Project/DeepLIIF_Dataset'
        Input_image = glob.glob(Input_dataset_path + "/*")
        print("Begin Split ...")
        for image_path in Input_image:
            (patch_path, patch_extname) = os.path.split(image_path)
            (patch_name, extension) = os.path.splitext(patch_extname)

            pending_img = cv2.imread(image_path)

            if 'neg' in image_path:
                Single_Negative_pic = Negative_image_path + '/' + '%s.png' % patch_name
                CD_pic = CD_img_save_path + '/' + '%s_posNeg.png' % patch_name
                CD_img = np.zeros((pending_img[0:512, 512:1024].shape[0], pending_img[0:512, 512:1024].shape[1], 1), np.uint8)


            else:
                Single_Positive_pic = Positive_image_path + '/' + '%s.png' % patch_name
                CD_pic = CD_img_save_path + '/' + '%s_Pos.png' % patch_name
                cv2.imwrite(Single_Positive_pic, pending_img[0:512, 0:512], [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损
                resize_CD_img = cv2.resize(pending_img[0:512, 512*4:512*5], (256, 256))
                cv2.imwrite(CD_pic, resize_CD_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损

            CD_pic = CD_img_save_path + '/' + '%s_Neg.png' % patch_name
            CD_img = np.zeros((256, 256, 3), np.uint8)
            Single_Negative_pic = Negative_image_path + '/' + '%s.png' % patch_name
            cv2.imwrite(Single_Negative_pic, pending_img[0:512, 512:1024], [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损
            cv2.imwrite(CD_pic, CD_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损

    Negative_mode = Negative_image_path.split("/")[-1]
    Positive_mode = Positive_image_path.split("/")[-1]

    write_txt(Negative_image_path, Negative_mode, Positive_image_path, Positive_mode, CD_img_save_path, seed_num, data_opt)
    txt_file.close()


if __name__ == '__main__':
    images_root = 'big_dataset'
    GHA_dir = 'CD_att'
    seed_num = '2021'
    main(images_root, GHA_dir, seed_num)


