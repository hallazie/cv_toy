import os
import numpy as np
from evaluation_scores import compute_scores, list_scores, get_images, all_fix_map


#saliency_folder = '/mnt/Databases/VDA/result/multi'
#saliency_folder = '/mnt/Databases/VDA/result/DHF1k/'
#saliency_folder = '/mnt/Databases/DHF1k/'
#density_folder = '/mnt/Databases/VDA/VIDEO_Saliency_database/annotation/maps/'
#fix_map_folder = '/mnt/Databases/VDA/VIDEO_Saliency_database/annotation/fixation/'

#saliency_folder = '/ssd/VDA/result/seo/'
#density_folder = '/ssd/rtavah1/VIDEO_Saliency_database/annotation/maps/'
#fix_map_folder = '/ssd/rtavah1/VIDEO_Saliency_database/annotation/fixation/'
#saliency_folder = '/ssd/FEVS'
#saliency_folder = '/ssd/VDA/result/deepVS/'
#saliency_folder = '/ssd/VDA/result/deepVS/'
#saliency_folder = '/ssd/VDA/result/video_from_multi/'

#saliency_folder = '/ssd/VDA/result/seo/'
#saliency_folder = '/ssd/VDA/result/awsd/'
#density_folder = '/ssd/rtavah1/VIDEO_Saliency_database/annotation/maps/'
#fix_map_folder = '/ssd/rtavah1/VIDEO_Saliency_database/annotation/fixation/'

saliency_folder = '/mnt/Databases/SALICON/results/model_bench/'
density_folder = '/mnt/Databases/SALICON/maps/val'
fix_map_folder = '/mnt/Databases/SALICON/fixmaps/val'


saliency_list = [os.path.join(saliency_folder, p) for p in os.listdir(saliency_folder)]
saliency_list = [os.path.join(saliency_folder, 'resnet_p_0.75')]

result = []

for sal in saliency_list:
    bname = os.path.basename(sal)
    dmapfolder = density_folder
    fmapfolder = fix_map_folder

    print("processing {}:".format(sal))
    scores, files = compute_scores(sal, dmapfolder, fmapfolder)
    eval_result = {'model': bname, 'scores': scores, 'files': files}
    result.append(eval_result)


np.save('resnet_pruned_075', result)


