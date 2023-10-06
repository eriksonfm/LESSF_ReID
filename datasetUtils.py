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

def load_from_Jadson(PATH, base_dir):
    
    images_names = []
    file = open(PATH, "r")
    file.readline()
    counter=0
    for line in file.readlines():
        img_name, pid_name = line.split(",")
        
        pid = int(pid_name[:-1]) # retirando o \n que sobrou da linha
        # o pid precisa attack/n_attack
        
        # camid = img_name.split("/")[5].split("_")[0]
        # o camid identifica o aparelho
        
        camid = counter
        counter +=1
        
        img_path = os.path.join(base_dir, img_name)
        images_names.append([img_path, pid, camid])
        
    images_names = np.array(images_names)
    return images_names


## Load target dataset
def load_dataset(dataset_name):
	
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
		base_name_dir = "/hadatasets/Synthetic-Realities/20-spoofing-mpad/2020-plosone-recod-mpad"

		train_images = load_from_Jadson("csvs/train_motog5.csv", base_name_dir) 
		gallery_images = load_from_Jadson("csvs/test_motog5.csv", base_name_dir) 
		queries_images = load_from_Jadson("csvs/val_motog5.csv", base_name_dir) 


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




