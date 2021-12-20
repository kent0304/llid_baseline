import torch
from torch import nn
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

# GPU対応
device = torch.device('cuda:0')

def image2vec(image_net, image_paths):
    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # stackはミニバッチに対応できる
    images = torch.stack([
        transformer(Image.open(image_path).convert('RGB'))
        for image_path in image_paths
    ])
    images = images.to(device)
    images = image_net(images)
    return images.cpu()

# def mydataset(dataset):
#     # resnet呼び出し
#     image_net = models.resnet50(pretrained=True)
#     image_net.fc = nn.Identity()
#     image_net.eval()
#     image_net = image_net.to(device)
#     data_num = len(dataset)
#     image_vec = torch.zeros((data_num, 2048))
#     batch_size = 512
#     with torch.no_grad():
#         for i in range(0, len(dataset), batch_size):
#             image_paths = [dataset[j][1] for j in range(i, len(dataset))[:batch_size]]
#             images = image2vec(image_net, image_paths)
#             image_vec[i:i + batch_size] = images

#             # if i >= 10*batch_size:
#             #     exit(0)
        
#     return image_vec

def get_image_paths():
    # 既に実験に利用している擬似データと実データのIDを取得
    MSCOCO_IMGFEAT_ROOT = "/mnt/LSTA6/data/tanaka/pdc/baseline/data/mscoco_imgfeats/"
    with open(MSCOCO_IMGFEAT_ROOT + 'train_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
        train_img_feats = pickle.load(f)
    with open(MSCOCO_IMGFEAT_ROOT + 'valid_synthetic_features_mscoco_half_crt.pkl', "rb") as f:
        val_img_feats = pickle.load(f)
    with open(MSCOCO_IMGFEAT_ROOT + 'raw_features.pkl', "rb") as f:
        raw_img_feats = pickle.load(f)
    train_image_ids = []
    val_image_ids = []
    raw_image_ids = []
    for img_datum in train_img_feats:
        img_id = img_datum['img_id']
        img_id = str(int(img_id.split('_')[-1]))
        train_image_ids.append(img_id)
    for img_datum in val_img_feats:
        img_id = img_datum['img_id']
        img_id = str(int(img_id.split('_')[-1]))
        val_image_ids.append(img_id)
    for img_datum in raw_img_feats:
        img_id = img_datum['img_id']
        img_id = str(int(img_id.split('_')[-1]))
        raw_image_ids.append(img_id)

    return train_image_ids, val_image_ids, raw_image_ids
    
    
def main():
    train_image_ids, val_image_ids, raw_image_ids = get_image_paths()
    # idをpathに変換
    MSCOCO_PATH = '/mnt/LSTA6/data/tanaka/mscoco/'

    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)
    # image_path = '/mnt/LSTA6/data/tanaka/mscoco/train2014/COCO_train2014_000000215625.jpg'
    
    train_result = []
    for image_id in tqdm(train_image_ids):
        img_d = {}
        img_d['img_id'] = image_id
        image_path = MSCOCO_PATH + 'train2017/' + (12-len(image_id))*'0' + image_id + '.jpg'
        image_feat = image2vec(image_net, [image_path])
        image_feat = image_feat.squeeze()
        img_d['global_feat'] = image_feat.detach().numpy()
        train_result.append(img_d)
    
    val_result = []
    for image_id in tqdm(val_image_ids):
        img_d = {}
        img_d['img_id'] = image_id
        image_path = MSCOCO_PATH + 'val2017/' + (12-len(image_id))*'0' + image_id + '.jpg'
        image_feat = image2vec(image_net, [image_path])
        image_feat = image_feat.squeeze()
        img_d['global_feat'] = image_feat.detach().numpy()
        val_result.append(img_d)

    raw_result = []
    for image_id in tqdm(raw_image_ids):
        img_d = {}
        img_d['img_id'] = image_id
        image_path = MSCOCO_PATH + 'train2017/' + (12-len(image_id))*'0' + image_id + '.jpg'
        image_feat = image2vec(image_net, [image_path])
        image_feat = image_feat.squeeze()
        img_d['global_feat'] = image_feat.detach().numpy()
        raw_result.append(img_d)

    with open('/mnt/LSTA6/data/tanaka/pdc/baseline/data/mscoco_imgfeats/global_train_synthetic_features_mscoco_half_crt.pkl', 'wb') as f:
        pickle.dump(train_result, f)
    with open('/mnt/LSTA6/data/tanaka/pdc/baseline/data/mscoco_imgfeats/global_valid_synthetic_features_mscoco_half_crt.pkl', 'wb') as f:
        pickle.dump(val_result, f)
    with open('/mnt/LSTA6/data/tanaka/pdc/baseline/data/mscoco_imgfeats/global_raw_features.pkl', 'wb') as f:
        pickle.dump(raw_result, f)



    

if '__main__' == __name__:
    main()