import os
import numpy as np

def load_set_from_market_duke(directory):
	
	images_names = []
	for filename in os.listdir(directory):
	    if filename.endswith(".jpg"):
	        camid = int(filename.split("_")[1][1])
	        pid = int(filename.split("_")[0])
	        if(pid != -1):
	            img_path = os.path.join(directory, filename)
	            images_names.append([img_path, pid, camid])
	            
	images_names = np.array(images_names)

	return images_names

def load_set_from_MSMT17(PATH, base_name):
	
	images_names = []
	train_file = open(PATH, "r")
	for line in train_file.readlines():
		img_name, pid_name = line.split(" ")

		pid = int(pid_name[:-1])
		camid = img_name.split("_")[2]
		
		img_path = os.path.join(base_name, img_name)
		images_names.append([img_path, pid, camid])

	images_names = np.array(images_names)
	return images_names

import cv2

def cria_arquivo_ruido(entrada, saida):
    img = cv2.imread(entrada)
    img = cv2.resize(img,dsize=(256, 128),interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_REAL_OUTPUT)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,0],fourier_shift[:,1]))
        
    #ruido = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    ruido = fourier_shift
    
    cv2.imwrite(saida,ruido)
    
def load_from_Jadson(PATH, base_dir, use_ruido=False):
    
    images_names = []
    file = open(PATH, "r")
    file.readline()
    counter=0
    for line in file.readlines():
        img_name, pid_name = line.split(",")
        
        pid = int(pid_name[:-1]) # retirando o \n que sobrou da linha
        # o pid precisa ser attack/n_attack
        
        # camid = img_name.split("/")[5].split("_")[0]
        # o camid identifica o aparelho
        
        camid = counter
        counter += 1
        
        work_dir = base_dir+'/'
        if use_ruido == True:
            novo_dir = "fft"
            img_name_nova = novo_dir + '/' + img_name
            parts = img_name_nova.split('/')
            parts.pop()
            novo_dir = '/'.join(parts)
            work_dir = "/work/emorais/"
            if not os.path.exists(work_dir+img_name_nova):
                os.makedirs(work_dir+novo_dir, exist_ok=True)
                cria_arquivo_ruido(entrada=base_dir+'/'+img_name, saida=work_dir+img_name_nova)
            img_name = img_name_nova
            
        img_path = os.path.join(work_dir, img_name)
        images_names.append([img_path, pid, camid])
        
    images_names = np.array(images_names)
    return images_names


## Load target dataset
def load_dataset(dataset_name, use_ruido=False):
	
	if dataset_name == "Market":
	
		train_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/Market-1501-v15.09.15/bounding_box_train")
		gallery_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/market1501/Market-1501-v15.09.15/bounding_box_test")
		queries_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/market1501/Market-1501-v15.09.15/query")

	elif dataset_name == "Duke":

		train_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/DukeMTMC-reID/bounding_box_train")
		gallery_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/DukeMTMC-reID/bounding_box_test")
		queries_images = load_set_from_market_duke("/hadatasets/ReID_Datasets/DukeMTMC-reID/query")

	elif dataset_name == "MSMT17":

		base_name_train = "/hadatasets/ReID_Datasets/MSMT17_V2/mask_train_v2"
		train_images = load_set_from_MSMT17("/hadatasets/ReID_Datasets/MSMT17_V2/list_train_uda.txt", base_name_train)

		base_name_test = "/hadatasets/ReID_Datasets/MSMT17_V2/mask_test_v2"
		gallery_images = load_set_from_MSMT17("/hadatasets/ReID_Datasets/MSMT17_V2/list_gallery.txt", base_name_test)
		queries_images = load_set_from_MSMT17("/hadatasets/ReID_Datasets/MSMT17_V2/list_query.txt", base_name_test)
  
	elif dataset_name == "Jadson":
		base_name_dir = "/hadatasets/Synthetic-Realities/20-spoofing-mpad/2020-plosone-recod-mpad/crops/"
	
		train_images   = load_from_Jadson("csvs/mpad_crop_just_motog_train.csv", base_name_dir, use_ruido) 
		gallery_images = load_from_Jadson("csvs/mpad_crop_just_motog_test.csv",  base_name_dir, use_ruido) 
		queries_images = load_from_Jadson("csvs/mpad_crop_just_motog_val.csv",   base_name_dir, use_ruido) 


	return train_images, gallery_images, queries_images

def load_text_dataset(base_dir, authors_set):

	training_txtfile = open(os.path.join(authors_set, "training_tweets.txt"), "r")
	query_txtfile = open(os.path.join(authors_set, "query_tweets.txt"), "r")
	gallery_txtfile = open(os.path.join(authors_set, "gallery_tweets.txt"), "r")

	train_text = []
	query_text = []
	gallery_text = []

	for sample in training_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		train_text.append([full_path, author_id, tweet_id])

	train_text = np.array(train_text)

	for sample in query_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		query_text.append([full_path, author_id, tweet_id])

	for sample in gallery_txtfile.readlines():
		author_id, tweet_id = sample[:-1].split(" ")
		full_path = os.path.join(base_dir, author_id, "tweets.json")
		gallery_text.append([full_path, author_id, tweet_id])

	query_text = np.array(query_text)
	gallery_text = np.array(gallery_text)

	return train_text, gallery_text, query_text




