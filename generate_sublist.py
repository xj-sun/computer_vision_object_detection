import os
import numpy as np
import h5py
import random
import linecache


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def dir2list(path,sub_list_file):
    if os.path.exists(sub_list_file):
        fp = open(sub_list_file, 'r')
        sublines = fp.readlines()
        sub_names = []
        for subline in sublines:
            sub_info = subline.replace('\n', '').split(',')
            sub_names.append(sub_info)
        fp.close()
        return sub_names
    else:
        fp = open(sub_list_file, 'w')
        img_root_dir = os.path.join(path)
        subs = os.listdir(img_root_dir)
        subs.sort()
        sub_names = []
        for image in subs:
        # for sub in subs:
            # sub_dir = os.path.join(img_root_dir,sub)
            # views = os.listdir(sub_dir)
            # views = ['view3']

            # views.sort()
            # for view in views:
                # view_dir = os.path.join(sub_dir,view)
            image_index = image.split('.')[0]
            seg_name = image_index + '_hm.jpg'
            # seg_dir = img_root_dir.replace('train','lmk_hm')

            image_path = os.path.join(img_root_dir, image)
            if 'train' in img_root_dir:
                seg_path = img_root_dir.replace('train', 'lmk_hm')
            if 'validation' in img_root_dir:
                seg_path = img_root_dir.replace('validation', 'lmk_hm')
            seg_path = os.path.join(seg_path,seg_name)
                # if not os.path.exists(seg_dir):
                #     test_view_dir = os.path.join(sub_dir+'_mask',view)
                #     test_seg_dir = test_view_dir.replace('/img/', '/seg/')
                #     if os.path.exists(test_seg_dir):
                #         seg_dir = test_seg_dir
                #     else:
                #         test_view_dir = os.path.join(sub_dir, view)
                #         test_seg_dir = test_view_dir.replace('/img/', '/seg/')
                #         test_seg_dir = test_seg_dir.replace('img', 'label')
                #         seg_dir = test_seg_dir

                # for slice in slices:
                    # subinfo = (sub,view,slice,view_dir,seg_dir)
                    # sub_names.append(subinfo)
            line = "%s,%s,%s"%(image, image_path, seg_path)
            fp.write(line + "\n")
        fp.close()

