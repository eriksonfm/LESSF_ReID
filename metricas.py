
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from datasetUtils import load_from_Jadson
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

########## DADOS GLOBAIS ##########

labels_teste = ''
labels_validation = ''

features_teste = ''
features_validation = ''

base_name_dir = "/hadatasets/Synthetic-Realities/20-spoofing-mpad/2020-plosone-recod-mpad"

def calc_predito(clusters, features, labels_ground_truth):
    
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(features)
    labels_kmt = kmeans.labels_

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

def medidas(GT, predito, k=0, lambda_hard=0,idx=0, grupo='test'):
    y_true = GT
    y_pred = predito
    confusion = confusion_matrix(GT, predito,normalize='true')
    
    
    disp = ConfusionMatrixDisplay(confusion)
    disp.plot()
    plt.savefig(f'resultados/MC_{k}_{lambda_hard}_{idx}_{grupo}.png')
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
    f1_score = f1_score(y_true, y_pred)
    
    TP = confusion[0,0]
    FN = confusion[0,1]
    FP = confusion[1,0]
    TN = confusion[1,1]
    
    # # Calcular o APCER
    # APCER = FN / (TP + FN)
         
    # # Calcular o BPCER
    # BPCER = FP / (TN + FP)
    
    return [accuracy, precision, recall, f1_score], ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE'] #incluir APCER e BPCER
    
def desenha_metricas(GT, features, labels_ground_truth, k=0, lambda_hard=0, idx=0, grupo='test', show_all=True):
        
    predito = calc_predito(clusters=2,features=features,labels_ground_truth=labels_ground_truth)
    metricas, rotulos = medidas(GT=GT,predito=predito, k=k, lambda_hard=lambda_hard, idx=idx, grupo=grupo)
    
    # print(GT)
    # print(predito)

    accuracy  = metricas[0]
    precision = metricas[1]
    recall    = metricas[2]
    f1_score  = metricas[3]
    #APCER     = metricas[4]
    #BPCER     = metricas[5]


    # print(f'Acurácia: {accuracy}')
    # print(f'Precisão: {precision}')
    # print(f'Recall: {recall}')
    # print(f'F1-Score: {f1_score}')
    # print(f'APCER: {APCER}')
    # print(f'BPCER: {BPCER}') 

    # Criar o gráfico de barras
    plt.bar(rotulos, metricas)

    # Adicionar rótulos e título
    plt.xlabel("Métricas")
    plt.ylabel("Valores")
    plt.title("Comparação das métricas")

    # Exibir o gráfico
    if (show_all == True):
        plt.show()
    
    plt.savefig(f'resultados/grafico_{k}_{lambda_hard}_{idx}_{grupo}.png')
    plt.close()
    return rotulos, metricas


def metricas(k, lambda_hard):
    
    #carregando labels

    labels_teste = np.load("resultados/labels_test_ruido.npy", allow_pickle=True).astype(int)
    labels_validation = np.load("resultados/labels_validation_ruido.npy", allow_pickle=True).astype(int)

    #print("tamanho de labels_teste: " + str(len(labels_teste)))
    #print("tamanho de labels_validation: " + str(len(labels_validation)))
    #corregando dados

    features_teste = torch.load("resultados/test_ruido.pt")
    features_validation = torch.load("resultados/validation_ruido.pt")

    #print("tamanho de features_teste: " + str(features_teste.size()))
    #print("tamanho de features_validation: " + str(features_validation.size()))

    # executando com conjunto de testes
    GT = load_from_Jadson("csvs/test_motog5.csv", base_name_dir, True)
    GT = np.array([ int(item[1]) for item in GT])
    
    tentativas =10

    metricas_t = []
    for i in range(tentativas):
        rotulos_t, metricas = desenha_metricas(GT=GT, features=features_teste, labels_ground_truth=labels_teste, k=k, lambda_hard=lambda_hard,idx=i, grupo='test', show_all=False)
        metricas_t.append(metricas)
    metricas_t = np.array(metricas_t)
    # print(metricas_t)


    # executando com conjunto de validação
    GT = load_from_Jadson("csvs/val_motog5.csv", base_name_dir, True)
    GT = np.array([ int(item[1]) for item in GT])
    metricas_v = []
    for i in range(tentativas):
        rotulos_v, metricas = desenha_metricas(GT=GT, features=features_validation, labels_ground_truth=labels_validation, k=k, lambda_hard=lambda_hard,idx=i, grupo='valid', show_all=False)
        metricas_v.append(metricas)
    metricas_v = np.array(metricas_v)
    # print(metricas_v)
    
    return metricas_t, metricas_v




