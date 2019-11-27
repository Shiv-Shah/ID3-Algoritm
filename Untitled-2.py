import numpy as np
import pandas as pd
import math as m

def GetEntropy(data):
    elem,counts = np.unique(data,return_counts=True)
    entropy = 0
   
    for i in range(len(elem)):
        prob = counts[i]/len(data)
        entropy += (0 - prob)*np.log2(counts[i]/len(data))
        print("-(", counts[i], "/", len(data), ") * log_2(", counts[i], "/", len(data), ") =", entropy)
	    
    #print("data:", data , " ", "Entropy:", entropy ,"\n")
    return entropy


    



def Inf_Gain(cond,attr,tar_name = "Action"):
    in_gain = GetEntropy(data[tar_name])
    print ("\nEntropy = ", in_gain, "for " ,cond, "With attribute\n", attr, "\n")
    vals,counts = np.unique(data[attr],return_counts=True)
    weight_ent = np.sum([(counts[i]/np.sum(counts))*GetEntropy(data.where(data[attr]==vals[i]).dropna()[tar_name]) for i in range(len(vals))])
    print("weighted entropy is", weight_ent, "\n")
    print("Information Gaim is", in_gain," - ", weight_ent, " = ", in_gain-weight_ent, "\n")
    in_gain -= weight_ent
    print("\nInformation Gain is:",attr, " ", tar_name, " ", in_gain)
    return in_gain


def ChooseHead(data,features,curr_feat, target_attribute_name):
    if len(data[curr_feat].value_counts()) == 1:
        print("\nWe have a pure subset. Leaf Node is creates\n", curr_feat, "for", target_attribute_name,"\n")
        print(data[curr_feat], "this is the data values")
        return 
    best_feat = None
    best_feat_ig = 0
    for feature in features:
        item_values = Inf_Gain(data,feature,target_attribute_name) 
        if item_values >= best_feat_ig:
            best_feat_ig = item_values
            best_feat = feature
    print("\nBest info gain = ",best_feat, "With the value of\n", best_feat_ig,"\n")
    features.remove(best_feat)
    for u_val in data[best_feat].unique():
        ChooseHead(data[data[best_feat] == u_val], features,best_feat,target_attribute_name)
   

        
    
    





if __name__ == '__main__':
    data = pd.read_csv("datasetID3.csv")
    print(data,"\n")
    # dp = Node(data,None,None)
    #features = data[0,-1]
    target_att = data.columns.values[-1]
    features = data.columns.values[:-1].tolist()
    
    ChooseHead(data,features,target_att, target_att)

    #Set_Entropy = Entropy(data[y_label]) #get entropy of our class column
    


            
        
    

    



