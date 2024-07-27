import glob
import os.path
import cv2
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    # GT和dehazed图像所在的路径
    #res_path = './psnrssim/results/Haze1k_thin'
    #gt_path = './psnrssim/GT/Haze1k_thin'
    res_path = 'D:/code/MSB/psnrssim/GT/UAV-Rain1k'
    gt_path = 'D:/code/MSB/psnrssim/results/UAV-Rain1k'

    psnr_all = []
    ssim_all = []
    img_list = sorted(glob.glob(gt_path + '/*'))

    for i, img_path in enumerate(img_list):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_gt = cv2.imread(img_path)
        img_res = cv2.imread(os.path.join(res_path, img_name + '.png'))

        psnr_rgb = cv2.PSNR(img_gt, img_res)
        ssim_rgb = ssim(img_gt, img_res, multichannel=True, channel_axis=2)

        print('{:3d} - {} \tPSNR: {:.4f} dB, \tSSIM: {:.4f}'.format(i+1, img_name, psnr_rgb, ssim_rgb))
        psnr_all.append(psnr_rgb)
        ssim_all.append(ssim_rgb)
        single_info = '{},PSNR,{:.4f}, SSIM,{:.4f}'

    Mean_format = 'Mean_PSNR: {:.4f}, Mean_SSIM: {:.4f}'
    Mean_PSNR = sum(psnr_all) / len(psnr_all)
    Mean_SSIM = sum(ssim_all) / len(ssim_all)
    print(Mean_format.format(Mean_PSNR, Mean_SSIM))