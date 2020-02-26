import torch
from torch.utils import data
import shutil
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
from tqdm import tqdm
import time
from config import Config
import numpy as np

def train(net, epoch, dataLoader, optimizer, trainF, config):
	net.train()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	epoch_loss = 0.0
	dataprocess = tqdm(dataLoader)
	criterion = Loss()

	for i, (img, gt_score, gt_geo, ignored_map) in enumerate(dataprocess):
		# start_time = time.time()
		img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
		optimizer.zero_grad()
		pred_score, pred_geo = net(img)
		loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
		
		epoch_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		dataprocess.set_description_str("epoch:{}".format(epoch))
		dataprocess.set_postfix_str("loss:{:.4f}".format(loss.item()))
	trainF.write("Epoch:{}, loss is {:.4f} \n".format(epoch, epoch_loss / len(dataLoader)))
	trainF.flush()


def test(net, epoch, dataLoader, testF, config):
    net.eval()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0.0
    dataprocess = tqdm(dataLoader)
	criterion = Loss()

	for i, (img, gt_score, gt_geo, ignored_map) in enumerate(dataprocess):
		img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
		pred_score, pred_geo = net(img)
		loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
		epoch_loss += loss.item()
		dataprocess.set_description_str("epoch:{}".format(epoch))
		dataprocess.set_postfix_str("loss:{:.4f}".format(loss.item()))
	testF.write("Epoch:{}, loss is {:.4f} \n".format(epoch, epoch_loss / len(dataLoader)))
	testF.flush()


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device_list = [0]

def main():
	config = Config()

	if os.path.exists(config.SAVE_PATH):
		shutil.rmtree(config.SAVE_PATH)
	os.makedirs(config.SAVE_PATH, exist_ok=True)

	trainF = open(os.path.join(config.SAVE_PATH, "train.csv"), 'w')
	testF = open(os.path.join(config.SAVE_PATH, "test.csv"), 'w')

	train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	val_img_path = os.path.abspath('../ICDAR_2015/test_img')
	val_gt_path  = os.path.abspath('../ICDAR_2015/test_gt')

	kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

	train_dataset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH*len(device_list), \
									shuffle=True, drop_last=True, **kwargs)

	val_dataset = custom_dataset(val_img_path, val_gt_path)
	val_loader = data.DataLoader(val_dataset, batch_size=config.TRAIN_BATCH*len(device_list), \
									shuffle=True, drop_last=True, **kwargs)

	net = EAST()

	if torch.cuda.is_available():
		net = net.cuda(device=device_list[0])
		net = torch.nn.DataParallel(net, device_ids=device_list)

	optimizer = torch.optim.Adam(net.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)

	for epoch in range(config.EPOCHS):
		train(net, epoch, train_loader, optimizer, trainF, config)
		test(net, epoch, val_loader, testF, config)
		if epoch != 0 and epoch % config.SAVE_INTERVAL == 0:
			torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), config.SAVE_PATH, "laneNet{}.pth.tar".format(epoch)))
	trainF.close()
	testF.close()
	torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(),  config.SAVE_PATH, "finalNet.pth.tar"))


if __name__ == '__main__':
	main()


	# pths_path      = './pths'
	# batch_size     = 24 
	# lr             = 1e-3
	# num_workers    = 4
	# epoch_iter     = 600
	# save_interval  = 5