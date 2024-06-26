
import torch
import numpy as np
from sklearn.cluster import *
import matplotlib.pyplot as plt
from datasetUtils import load_from_Jadson
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import silhouette_score

########## DADOS GLOBAIS ##########

labels_teste = ''
labels_validation = ''

features_teste = ''
features_validation = ''

base_name_dir = "/hadatasets/Synthetic-Realities/20-spoofing-mpad/2020-plosone-recod-mpad"

distancias_t = []
distancias_v = []
s_models_t   = []
s_models_v   = []

def calc_predito(clusters, features, labels_ground_truth, grupo, modelo):
    
    agg_clustering = AgglomerativeClustering(n_clusters=clusters, metric='euclidean', linkage='ward')
        
    # features = features/torch.norm(features, dim=1, keepdim=True)
    # distance_matrix = 1.0 - torch.mm(features, features.T)
    # 
    # features = features/torch.norm(features, dim=1, keepdim=True)
    # distance_matrix = 1.0 - torch.mm(features, features.T)
    
    # distance_ensemble = (distance_R50 + distance_OSN + distance_DEN)/3

    # Salvar as tres, carregar as 3 e colocar no próximo argumento
    if modelo == "mean":
        if grupo == "test":
            array = np.array(distancias_t)
            s_models = s_models_t
        elif grupo == "valid":
            array = np.array(distancias_v)
            s_models = s_models_v
                
        # media = np.mean(array, axis=0)
        media = np.average(array, 0, s_models) # ponderada pelas medidas de silhouette
        distance_matrix = media
        
    else:
        features = features/torch.norm(features, dim=1, keepdim=True)
        distance_matrix = 1.0 - torch.mm(features, features.T)
        # colocar em um vetor global para fazer a mistura depois
        
        if grupo == 'test':
            distancias_t.append(distance_matrix)
        elif grupo == 'valid':
            distancias_v.append(distance_matrix)
        distance_matrix = distance_matrix.numpy()
        
    agg_clustering.fit(distance_matrix) 
    labels_kmt = agg_clustering.labels_
    
    if modelo == "mean":
        if grupo == 'test':
            s_models = s_models_t
        elif grupo == 'valid':
            s_models = s_models_v
            
        s_model = np.array(s_models)
        s_model = np.mean (s_model, axis=0)
        
    else:
        s_model = (silhouette_score(distance_matrix, labels_kmt, metric='euclidean') +1)/2.0 # 0 <= s_model <= 1
        
    if grupo == 'test':
        s_models_t.append(s_model)
    elif grupo == 'valid': 
        s_models_v.append(s_model)
         
    # acumular s_model para cada modelo: olhar para melhor, media ponderada (precisa estar entre 0 e 1)

    predito = np.zeros(len(labels_ground_truth), dtype=int)
    zero_idx = np.where(labels_kmt==0)[0]
    one_idx  = np.where(labels_kmt==1)[0]
    GT_zero = labels_ground_truth[zero_idx]
    labels,frequecia = np.unique(GT_zero, return_counts=True)
        
    if len(labels)==2 :
        if frequecia[0] > frequecia[1] : 
            predito[zero_idx] = labels[0]
            predito[one_idx]  = labels[1]
            
        else: 
            predito[zero_idx] = labels[1]
            predito[one_idx]  = labels[0]
                     
    return predito 

def medidas(GT, predito, modelo, k=0, lambda_hard=0,idx=0, grupo='test'):
    y_true = GT
    y_pred = predito
    confusion = confusion_matrix(GT, predito,normalize='true')
    
    
    disp = ConfusionMatrixDisplay(confusion)
    disp.plot()
    plt.savefig(f'resultados/MC_{k}_{lambda_hard}_{idx}_{modelo}_{grupo}.png')
    plt.close()

    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # calculando as métricas iniciais
    
    # # Calcular a acurácia (Accuracy)
    accuracy = accuracy_score(y_true, y_pred)
        
    # Calcular a precisão (Precision)
    precision = precision_score(y_true, y_pred, zero_division="warn")
    
    # Calcular o recall (Sensibilidade)
    recall = recall_score(y_true, y_pred)
     
    # Calcular o F1-Score
    f1score = f1_score(y_true, y_pred)
    
    TP = confusion[0,0]
    FN = confusion[0,1]
    FP = confusion[1,0]
    TN = confusion[1,1]
    
    
    far = FP / (TN+FP) #apcer
    frr = FN / (FN+TP) #
        
    # # Calcular o APCER (teste)
    # APCER = FP / (TN + FP)
    apcer = far
         
    # # Calcular o BPCER (teste)
    # BPCER = FN / (TP + FN)
    bpcer = frr
    
    # # Calcular o ACER
    # ACER = (APCER + BPCER) /2
    acer = (apcer+bpcer)/2.0
    
    #fonte: https://sites.google.com/view/face-anti-spoofing-challenge/evaluation
    
    
    if grupo == 'test':
        s_models = s_models_t
    elif grupo == 'valid':
        s_models = s_models_v
        
    if modelo =="mean":
        silhouette = np.array(s_models)
        silhouette = np.mean(silhouette)
        
    else:
        silhouette = s_models[-1]
    
    return [
        accuracy
        ,precision
        ,recall
        ,f1score
        ,apcer
        ,bpcer
        ,acer
        ,silhouette 
            ],['ACCURACY'
               ,'PRECISION'
               ,'RECALL'
               ,'F1_SCORE'
               ,'APCER'
               ,'BPCER'
               ,'ACER'
               ,'SILHOUETTE'
               ]
    
def desenha_metricas(GT, features, labels_ground_truth, modelo, k=0, lambda_hard=0, idx=0, grupo='test', show_all=True):
        
    predito = calc_predito(clusters=2,features=features,labels_ground_truth=labels_ground_truth, grupo=grupo, modelo=modelo)
    metricas, rotulos = medidas(GT=GT,predito=predito, modelo=modelo, k=k, lambda_hard=lambda_hard, idx=idx, grupo=grupo)
 
    #accuracy  = metricas[0]
    #precision = metricas[1]
    #recall    = metricas[2]
    #f1_score  = metricas[3]
    #EER       = metricas[4]
    #HTER      = metricas[5]
    #APCER     = metricas[6]
    #BPCER     = metricas[7]

    # Criar o gráfico de barras
    plt.bar(rotulos, metricas)
    plt.xticks(rotation=30, fontsize=5)

    # Adicionar rótulos e título
    plt.xlabel('Métricas')
    plt.ylabel("Valores")
    plt.title("Comparação das métricas")

    # Exibir o gráfico
    if (show_all == True):
        plt.show()
    
    plt.savefig(f'resultados/grafico_{k}_{lambda_hard}_{idx}_{modelo}_{grupo}.png')
    plt.close()
    return rotulos, metricas


def metricas(k, lambda_hard, modelo):
    
    #carregando labels

    labels_teste = np.load("resultados/labels_test_ruido.npy", allow_pickle=True).astype(int)
    labels_validation = np.load("resultados/labels_validation_ruido.npy", allow_pickle=True).astype(int)

    #corregando dados
    if (modelo == 'mean'):
        features_teste = ""
        features_validation = ""
    else:
        features_teste = torch.load("resultados/test_ruido_" + modelo + ".pt")
        features_validation = torch.load("resultados/validation_ruido_" + modelo +".pt")

    # executando com conjunto de testes
    GT = load_from_Jadson("csvs/test_motog5.csv", base_name_dir, True)
    GT = np.array([ int(item[1]) for item in GT])
    
    tentativas =1
    
    metricas_t = []
    for i in range(tentativas):
        rotulos_t, metricas = desenha_metricas(GT=GT, features=features_teste, labels_ground_truth=labels_teste, modelo=modelo, k=k, lambda_hard=lambda_hard,idx=i, grupo='test', show_all=False)
        metricas_t.append(metricas)
    metricas_t = np.array(metricas_t)

    # executando com conjunto de validação
    GT = load_from_Jadson("csvs/val_motog5.csv", base_name_dir, True)
    GT = np.array([ int(item[1]) for item in GT])
    
    metricas_v = []
    for i in range(tentativas):
        rotulos_v, metricas = desenha_metricas(GT=GT, features=features_validation, labels_ground_truth=labels_validation, modelo=modelo, k=k, lambda_hard=lambda_hard,idx=i, grupo='valid', show_all=False)
        metricas_v.append(metricas)
    metricas_v = np.array(metricas_v)
    
    return metricas_t, metricas_v, rotulos_t, rotulos_v




