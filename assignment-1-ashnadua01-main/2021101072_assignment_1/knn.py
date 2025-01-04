import pandas as pd
import numpy as np
from numpy.linalg import norm
import sys
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = np.load('data.npy', allow_pickle=True)
data_df = pd.DataFrame(data)

class KNN:
    def __init__(self, k=3, distance_metric="Euclidean", encoder="ResNet", split=0.2):
        self.k = k
        self.distance_metric=distance_metric
        self.encoder = encoder
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.split = split
        self.set_data()
        
    def set_data(self):
        if(self.encoder == 'ResNet'):
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(data_df[1], data_df[3], test_size=self.split, random_state=42, shuffle=True)
        elif(self.encoder == 'VIT'):
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(data_df[2], data_df[3], test_size=self.split, random_state=42, shuffle=True)
                    
    def calc_distance_euclidean(self, x1, x2):
        dist = np.sqrt(np.sum((x1 - x2) ** 2))
        return dist
    
    def calc_distance_manhattan(self, x1, x2):
        dist = np.sum(np.abs(x1 - x2))
        return dist
    
    def calc_distance_cosine(self, x1, x2):
        dot_product = np.dot(x1.reshape(-1), x2.reshape(-1))
        norm_x1 = norm(x1)
        norm_x2 = norm(x2)

        if norm_x1 == 0 or norm_x2 == 0:
            return 1

        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
    
    def predict(self, x, x_train, y_train):
        distances = []
        label_count = dict()
        label_weights = dict()
    
        for point, label in zip(x_train, y_train):
            if self.distance_metric == "Euclidean":
                dist = self.calc_distance_euclidean(x, point)
            elif self.distance_metric == "Manhattan":
                dist = self.calc_distance_manhattan(x, point)
            elif self.distance_metric == "Cosine":
                dist = self.calc_distance_cosine(x, point)
            distances.append((dist, label))
    
        distances.sort(key=lambda x: x[0])
    
        for i in range(0, self.k):
            curr_dist, curr_label = distances[i]
            if curr_label in label_count:
                label_count[curr_label] += 1
            else:
                label_count[curr_label] = 1
        
            weight = 1 / (curr_dist + 1e-6)
            if curr_label in label_weights:
                label_weights[curr_label] += weight
            else:
                label_weights[curr_label] = weight
    
        max_count = max(label_count.values())
        potential_labels = []
        for label, count in label_count.items():
            if count == max_count:
                potential_labels.append(label)
    
        if len(potential_labels) == 1:
            return potential_labels[0]
        else:
            max_weight_label = max(potential_labels, key=lambda label: label_weights.get(label, 0))
            return max_weight_label
    
    def inference(self, x):
        print(self.predict(x))
    
    def evaluate_metrics(self, x_train, y_train, x_val, y_val):
        predictions = []
        for x in x_val:
            predictions.append(self.predict(x, x_train, y_train))
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, average='weighted', zero_division=1)
        recall = recall_score(y_val, predictions, average='weighted', zero_division=1)
        f1_micro = f1_score(y_val, predictions, average='micro')
        f1_macro = f1_score(y_val, predictions, average='macro')
        f1_weighted = f1_score(y_val, predictions, average='weighted')
        
        return accuracy, precision, recall, f1_micro, f1_macro, f1_weighted
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python knn.py <path_to_test_data>")
        sys.exit(1)
    
    test_data_path = sys.argv[1]
    
    test_data = np.load(test_data_path, allow_pickle=True)
    test_data_df = pd.DataFrame(test_data)

    x_test_resnet = test_data_df[1]
    x_test_vit = test_data_df[2]
    y_test = test_data_df[3]

    knn_model_vit = KNN(k=12, distance_metric="Manhattan", encoder="VIT")
    knn_model_resnet = KNN(k=15, distance_metric="Manhattan", encoder="ResNet")

    accuracy_vit, precision_vit, recall_vit, f1_micro_vit, f1_macro_vit, f1_weighted_vit = knn_model_vit.evaluate_metrics(knn_model_vit.x_train, knn_model_vit.y_train, x_test_vit, y_test)
    accuracy_resnet, precision_resnet, recall_resnet, f1_micro_resnet, f1_macro_resnet, f1_weighted_resnet = knn_model_resnet.evaluate_metrics(knn_model_resnet.x_train, knn_model_resnet.y_train, x_test_resnet, y_test)

    metrics_dict = {}
    metrics_dict["VIT"] = {
        "Accuracy": accuracy_vit,
        "Precision": precision_vit,
        "Recall": recall_vit,
        "F1_Micro": f1_micro_vit,
        "F1_Macro": f1_macro_vit,
        "F1_Weighted": f1_weighted_vit
    }

    metrics_dict["ResNet"] = {
        "Accuracy": accuracy_resnet,
        "Precision": precision_resnet,
        "Recall": recall_resnet,
        "F1_Micro": f1_micro_resnet,
        "F1_Macro": f1_macro_resnet,
        "F1_Weighted": f1_weighted_resnet
    }

    table_data = []
    for encoder, metrics in metrics_dict.items():
        accuracy = metrics["Accuracy"]
        recall = metrics["Recall"]
        precision = metrics["Precision"]
        f1_micro = metrics["F1_Micro"]
        f1_macro = metrics["F1_Macro"]
        f1_weighted = metrics["F1_Weighted"]

        table_data.append([encoder, accuracy, precision, recall, f1_micro, f1_macro])
    
    headers = ["Encoder", "Accuracy", "Precision","Recall", "F1 Micro", "F1 Macro", "F1 Weighted"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
