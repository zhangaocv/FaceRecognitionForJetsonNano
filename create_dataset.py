# -*-coding: utf-8 -*-
import torch
import numpy as np
import onnxruntime
import cv2
import os

from utils import file_processing
from utils.general import normalize_img, post_process
from utils.extract_face import extract

def get_face_embedding(files_list, names_list):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(onnxruntime.get_device()))

    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    face_detect_session = onnxruntime.InferenceSession("checkpoints/yolov7-lite-s.onnx", None, providers=providers)
    face_recognition_session = onnxruntime.InferenceSession("checkpoints/20180402-114759-vggface2.onnx", None,
                                                                 providers=providers)

    embeddings = [] # 用于保存人脸特征数据库
    label_list = [] # 保存人脸label的名称，与embeddings一一对应
    for image_path, name in zip(files_list, names_list):
        print("processing image :{}".format(image_path))
        im = cv2.imread(image_path)
        x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        x_onnx = normalize_img(x, 0.0, 0.00392156862745098)
        detect_input_name = face_detect_session.get_inputs()[0].name
        detect_output = face_detect_session.run([], {detect_input_name: x_onnx})[0]
        batch_boxes, prob, _ = post_process(torch.from_numpy(detect_output).to(device), x_onnx, im)
        if len(batch_boxes) != 0:
            x_aligned = extract(x, batch_boxes, keep_all=True)
            recog_input_name = face_recognition_session.get_inputs()[0].name
            pred_embeddings = face_recognition_session.run([], {recog_input_name: np.array(x_aligned[0].unsqueeze(0))})[0]

            # image_processing.show_image_boxes("image",image,bboxes)

            embeddings.append(pred_embeddings)
            # 可以选择保存image_list或者names_list作为人脸的标签
            # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
            # label_list.append(image_path)
            label_list.append(name)
    return embeddings,label_list

def create_face_embedding(dataset_path,out_emb_path,out_filename):

    files_list,names_list=file_processing.gen_files_labels(dataset_path,postfix=['*.jpg'])
    embeddings,label_list=get_face_embedding(files_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))

    embeddings = np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_list_data(out_filename, label_list, mode='w')

def create_face_embedding_for_bzl(dataset_path,out_emb_path,out_filename):

    image_list = file_processing.get_images_list(dataset_path, postfix=['*.jpg', '*.png'])
    names_list=[]
    for image_path in image_list:
        basename = os.path.basename(image_path)
        names = basename.split('_')[0]
        names_list.append(names)
    embeddings,label_list=get_face_embedding(image_list, names_list)
    print("label_list:{}".format(label_list))
    print("have {} label".format(len(label_list)))
    embeddings=np.asarray(embeddings)
    np.save(out_emb_path, embeddings)
    file_processing.write_data(out_filename, label_list, mode='w')

if __name__ == '__main__':
    dataset_path = 'data/images'
    out_emb_path = 'data/emb/faceEmbedding.npy'
    out_filename = 'data/emb/name.txt'
    create_face_embedding(dataset_path,out_emb_path, out_filename)
