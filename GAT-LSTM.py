import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import chardet
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  


def set_seed(seed=42):
    """
   
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
   


set_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  
    return result['encoding']



def load_data(water_quality_path, edge_list_path):
    if not os.path.exists(water_quality_path):
        raise FileNotFoundError(f"水质数据文件未找到: {water_quality_path}")
    if not os.path.exists(edge_list_path):
        raise FileNotFoundError(f"边列表文件未找到: {edge_list_path}")

   
    water_quality_encoding = detect_encoding(water_quality_path)
    edge_list_encoding = detect_encoding(edge_list_path)

    print(f"水质数据编码: {water_quality_encoding}")
    print(f"边列表编码: {edge_list_encoding}")

   
    water_quality_df = pd.read_csv(water_quality_path, encoding=water_quality_encoding, parse_dates=['监测时间'])

   
    edge_list_df = pd.read_csv(edge_list_path, encoding=edge_list_encoding)
    if not {'源节点', '目标节点', '权重'}.issubset(edge_list_df.columns):
        raise ValueError("边列表文件必须包含 '源节点', '目标节点', '权重' 三个表头。")

    return water_quality_df, edge_list_df



def preprocess_data(water_quality_df):
    scaler = StandardScaler()
   
    feature_cols = ['溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮']
    features = water_quality_df[feature_cols].values
    scaled_features = scaler.fit_transform(features)
    water_quality_df[feature_cols] = scaled_features
    return water_quality_df, scaler


def create_sequences_all_nodes(df, seq_length):
    """
    构建包含所有节点的时间序列数据。
    返回X: (num_samples, seq_length, num_nodes, num_features)
          y: (num_samples, num_nodes, num_features)
    """
    sequences = []
    targets = []
   
    timestamps = sorted(df['监测时间'].unique())
    num_nodes = df['node_id'].nunique()

    for i in range(len(timestamps) - seq_length):
        seq = []
        for t in range(i, i + seq_length):
            time_data = df[df['监测时间'] == timestamps[t]].sort_values('node_id')
            features = time_data[['溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮']].values  
            if features.shape[0] != num_nodes:
                raise ValueError(f"时间戳 {timestamps[t]} 的节点数量不匹配。")
            seq.append(features)
        target_time = i + seq_length
        target_data = df[df['监测时间'] == timestamps[target_time]].sort_values('node_id')[
            ['溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮']].values  
        if target_data.shape[0] != num_nodes:
            raise ValueError(f"时间戳 {timestamps[target_time]} 的节点数量不匹配。")
        sequences.append(seq)
        targets.append(target_data)
    return np.array(sequences), np.array(targets)





class WaterQualityDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array of shape (num_samples, seq_length, num_nodes, num_features)
        y: numpy array of shape (num_samples, num_nodes, num_features)
        """
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class GAT_LSTM(nn.Module):
    def __init__(self, num_features, hidden_gat, hidden_lstm, num_nodes, edge_index, edge_weight=None, dropout=0.3,
                 num_gat_layers=1, num_heads=8):
        super(GAT_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.gat_layers = nn.ModuleList()
       
        self.gat_layers.append(GATConv(num_features, hidden_gat // num_heads, heads=num_heads, dropout=dropout))
        for _ in range(num_gat_layers - 1):
           
            self.gat_layers.append(GATConv(hidden_gat, hidden_gat // num_heads, heads=num_heads, dropout=dropout))
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(hidden_gat * num_nodes, hidden_lstm, batch_first=True)
        self.fc = nn.Linear(hidden_lstm, num_nodes * num_features)
        self.edge_index = edge_index
        self.edge_weight = edge_weight  

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_length, num_nodes, num_features)
        """
        batch_size, seq_length, num_nodes, num_features = x.size()
        gat_out_seq = []
        for t in range(seq_length):
            
            x_t = x[:, t, :, :]  
           
            x_t = x_t.reshape(batch_size * num_nodes, num_features)  
           
            for gat in self.gat_layers:
                x_t = gat(x_t, self.edge_index, self.edge_weight)  
                x_t = self.elu(x_t)
                x_t = self.dropout(x_t)
           
            gat_out = x_t.reshape(batch_size,
                                  num_nodes * self.gat_layers[-1].out_channels * self.gat_layers[-1].heads)  
            gat_out_seq.append(gat_out)
        
        gat_out_seq = torch.stack(gat_out_seq, dim=1)  
        lstm_out, _ = self.lstm(gat_out_seq)  
        lstm_out = self.dropout(lstm_out)
       
        lstm_last = lstm_out[:, -1, :]  
        
        out = self.fc(lstm_last)  
        
        out = out.reshape(batch_size, self.num_nodes, -1)  
        return out




def worker_init_fn(worker_id):
    # 确保每个子进程的随机种子不同
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)



def main():
    # 文件路径
    water_quality_path = r'.csv'  
    edge_list_path = r'.csv'  

   
    water_quality_df, edge_list_df = load_data(water_quality_path, edge_list_path)
    print("数据加载完成。")

   
    water_quality_df = water_quality_df.ffill().bfill()

    water_quality_df, scaler = preprocess_data(water_quality_df)
    print("数据归一化完成。")

    
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler 已保存为 'scaler.pkl'。")

   
    num_nodes = water_quality_df['node_id'].nunique()
    feature_cols = ['溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮']
    num_features = len(feature_cols)  
    print(f"节点数: {num_nodes}, 特征数: {num_features}")

    
    selected_features = ['总磷', '总氮']
    selected_indices = [feature_cols.index(feat) for feat in selected_features]
    print(f"选择评估的特征: {selected_features}")

   
    SEQ_LENGTH = 10  
    X, y = create_sequences_all_nodes(water_quality_df, SEQ_LENGTH)
    print(f"所有节点的序列创建完成。X shape: {X.shape}, y shape: {y.shape}")

   
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
    print(
        f"数据集划分完成。训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    
    train_dataset = WaterQualityDataset(X_train, y_train)
    val_dataset = WaterQualityDataset(X_val, y_val)
    test_dataset = WaterQualityDataset(X_test, y_test)  

    BATCH_SIZE = 64

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn,
                              num_workers=4)  
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=worker_init_fn,
                            num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=worker_init_fn,
                             num_workers=4)  
    print("数据加载器创建完成。")

   
    edge_list = edge_list_df[['源节点', '目标节点', '权重']].values
   
    if edge_list[:, 0].max() >= num_nodes or edge_list[:, 1].max() >= num_nodes:
        raise ValueError("边列表中的节点编号超过实际节点数。请检查节点编号是否从0开始且连续。")

    edge_index = torch.tensor(edge_list[:, :2].T, dtype=torch.long).to(device)  
    edge_weight = torch.tensor(edge_list[:, 2], dtype=torch.float).to(device)  
    print(f"edge_index 形状: {edge_index.shape}")
    print(f"edge_weight 形状: {edge_weight.shape}")

    
    HIDDEN_GAT = 32  
    HIDDEN_LSTM = 128
    DROPOUT = 0.3
    NUM_GAT_LAYERS = 2  
    NUM_HEADS = 1  

    model = GAT_LSTM(num_features=num_features, hidden_gat=HIDDEN_GAT, hidden_lstm=HIDDEN_LSTM,
                    num_nodes=num_nodes, edge_index=edge_index, edge_weight=edge_weight, dropout=DROPOUT,
                    num_gat_layers=NUM_GAT_LAYERS, num_heads=NUM_HEADS)
    model = model.to(device)
    print("模型初始化完成。")

   
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("损失函数和优化器定义完成。")

  
    def train_epoch(model, dataloader, criterion, optimizer):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def evaluate(model, dataloader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate_metrics(model, dataloader, scaler):
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
              
                output = output.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                batch_y = batch_y.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
              
                for i in range(batch_y.shape[0]):
                    y_true.append(scaler.inverse_transform(batch_y[i]))
                    y_pred.append(scaler.inverse_transform(output[i]))
        y_true = np.array(y_true).reshape(-1, num_features)
        y_pred = np.array(y_pred).reshape(-1, num_features)

       
        y_true_selected = y_true[:, selected_indices]
        y_pred_selected = y_pred[:, selected_indices]

        mae = mean_absolute_error(y_true_selected, y_pred_selected)
        mse = mean_squared_error(y_true_selected, y_pred_selected)
        rmse = np.sqrt(mse)
        
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true_selected - y_pred_selected) / (y_true_selected + epsilon))) * 100
        r2 = r2_score(y_true_selected, y_pred_selected)
        return mae, rmse, mape, r2

    
    EPOCHS = 1000
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f'第 {epoch}/{EPOCHS} 轮, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')

       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("早停触发。")
                break

    print("训练完成。")

    
    
    model.load_state_dict(torch.load('best_model.pth'))

    
    final_val_loss = evaluate(model, val_loader, criterion)
    final_test_loss = evaluate(model, test_loader, criterion)  
    print(f'最佳验证损失: {final_val_loss:.4f}')
    print(f'测试集损失: {final_test_loss:.4f}')  

   
    train_mae, train_rmse, train_mape, train_r2 = evaluate_metrics(model, train_loader, scaler)
    val_mae, val_rmse, val_mape, val_r2 = evaluate_metrics(model, val_loader, scaler)
    test_mae, test_rmse, test_mape, test_r2 = evaluate_metrics(model, test_loader, scaler)  
    print(
        f'训练集 MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.4f}')
    print(
        f'验证集 MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}')
    print(
        f'测试集 MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}')  
    

   
    def inverse_transform_predictions(scaler, predictions):
        """
     
        """
      
        original_scale = scaler.inverse_transform(predictions)
        
        df = pd.DataFrame(original_scale, columns=feature_cols)
        return df

    def predict_sample(model, input_sequence, scaler):
        """
       
        """
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_sequence, dtype=torch.float).unsqueeze(0).to(
                device)  
            prediction = model(input_tensor)  
            
            prediction = prediction.detach().cpu().numpy().reshape(num_nodes, num_features)
       
        prediction_original = inverse_transform_predictions(scaler, prediction)
        return prediction_original

    
    sample_X = X_test[0]  # (seq_length, num_nodes, num_features)
    predicted_water_quality = predict_sample(model, sample_X, scaler)

   
    def visualize_predictions_scatter(y_true_val, y_pred_val, feature_names, node_id):
        """
        y_true_val: numpy array of shape (num_samples, num_features)
        y_pred_val: numpy array of shape (num_samples, num_features)
        feature_names: list of feature names
        node_id: int, the node to plot (0-12)
        """
        for feature_idx, feature in enumerate(feature_names):
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_val[:, feature_idx], y_pred_val[:, feature_idx], alpha=0.5, label='预测值 vs 实际值')
            max_val = max(y_true_val[:, feature_idx].max(), y_pred_val[:, feature_idx].max())
            min_val = min(y_true_val[:, feature_idx].min(), y_pred_val[:, feature_idx].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 y=x')
            plt.title(f'节点 {node_id} - {feature} 预测值 vs 实际值')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.legend()
            plt.grid(True)
            plt.show()

   
    def get_all_predictions(model, dataloader, scaler):
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
               
                output = output.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                batch_y = batch_y.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
               
                for i in range(batch_y.shape[0]):
                    y_true.append(scaler.inverse_transform(batch_y[i]))
                    y_pred.append(scaler.inverse_transform(output[i]))
        y_true = np.array(y_true).reshape(-1, num_features)
        y_pred = np.array(y_pred).reshape(-1, num_features)
        return y_true, y_pred

   
    y_true_val, y_pred_val = get_all_predictions(model, val_loader, scaler)
    y_true_test, y_pred_test = get_all_predictions(model, test_loader, scaler) 

   
    feature_names = selected_features  
    node_id_to_plot = 0  

    
    if node_id_to_plot < 0 or node_id_to_plot >= num_nodes:
        raise ValueError(f"node_id_to_plot ({node_id_to_plot}) 超出范围 (0-{num_nodes - 1})")

    for feature_idx, feature in enumerate(feature_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_test[:, selected_indices[feature_idx]], y_pred_test[:, selected_indices[feature_idx]], alpha=0.5, label='预测值 vs 实际值')
        max_val = max(y_true_test[:, selected_indices[feature_idx]].max(), y_pred_test[:, selected_indices[feature_idx]].max())
        min_val = min(y_true_test[:, selected_indices[feature_idx]].min(), y_pred_test[:, selected_indices[feature_idx]].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 y=x')
        plt.title(f'节点 {node_id_to_plot} - {feature} 预测值 vs 实际值 (测试集)')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def simulate_missing_nodes(X, missing_nodes):
        """
        
        """
        X_missing = X.copy()
        X_missing[:, :, missing_nodes, :] = 0  
        return X_missing

   
    def simulate_missing_nodes_and_evaluate(model, X_data, y_data, scaler, missing_nodes):
        """
        模拟输入数据中缺失某些节点的特征值，并评估对预测结果的影响。

        参数:
        - model: 训练好的模型
        - X_data: numpy array, 输入数据
        - y_data: numpy array, 真实数据
        - scaler: StandardScaler 对象
        - missing_nodes: list of int, 要模拟缺失的节点编号

        返回:
        - metrics_missing: dict, 包含 MAE, RMSE, MAPE, R²
        """
        # 模拟缺失数据
        X_missing = simulate_missing_nodes(X_data, missing_nodes)
       
        missing_dataset = WaterQualityDataset(X_missing, y_data)
        missing_loader = DataLoader(missing_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    worker_init_fn=worker_init_fn, num_workers=4)

        # 评估指标
        mae, rmse, mape, r2 = evaluate_metrics(model, missing_loader, scaler)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

    # 定义缺失数据影响可视化函数
    def visualize_missing_impact(model, X_data, y_data, scaler, missing_nodes, node_id_to_plot, dataset_type):
        """
        可视化缺失数据情况下的预测结果变化。

        参数:
        - model: 训练好的模型
        - X_data: numpy array, 输入数据
        - y_data: numpy array, 真实数据
        - scaler: StandardScaler 对象
        - missing_nodes: list of int, 要模拟缺失的节点编号
        - node_id_to_plot: int, 要绘制的节点编号
        - dataset_type: str, 数据集类型（'验证集' 或 '测试集'）
        """
        # 模拟缺失数据
        X_missing = simulate_missing_nodes(X_data, missing_nodes)
        # 创建数据集和数据加载器
        missing_dataset = WaterQualityDataset(X_missing, y_data)
        missing_loader = DataLoader(missing_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                    worker_init_fn=worker_init_fn, num_workers=4)

        # 获取真实值和预测值（完整数据）
        y_true_full, y_pred_full = get_all_predictions(model, DataLoader(WaterQualityDataset(X_data, y_data),
                                                                         batch_size=BATCH_SIZE, shuffle=False,
                                                                         worker_init_fn=worker_init_fn, num_workers=4),
                                                       scaler)

        # 获取真实值和预测值（缺失数据）
        y_true_missing, y_pred_missing = get_all_predictions(model, missing_loader, scaler)

        # 绘制指定节点的每个特征的实际值与预测值的散点图
        for feature_idx, feature in enumerate(selected_features):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(y_true_full[:, selected_indices[feature_idx]], y_pred_full[:, selected_indices[feature_idx]], alpha=0.3, label='完整数据')
            max_val = max(y_true_full[:, selected_indices[feature_idx]].max(), y_pred_full[:, selected_indices[feature_idx]].max())
            min_val = min(y_true_full[:, selected_indices[feature_idx]].min(), y_pred_full[:, selected_indices[feature_idx]].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 y=x')
            plt.title(f'完整数据 - 节点 {node_id_to_plot} - {feature} ({dataset_type})')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.scatter(y_true_missing[:, selected_indices[feature_idx]], y_pred_missing[:, selected_indices[feature_idx]], alpha=0.3, label='缺失数据')
            max_val_m = max(y_true_missing[:, selected_indices[feature_idx]].max(), y_pred_missing[:, selected_indices[feature_idx]].max())
            min_val_m = min(y_true_missing[:, selected_indices[feature_idx]].min(), y_pred_missing[:, selected_indices[feature_idx]].min())
            plt.plot([min_val_m, max_val_m], [min_val_m, max_val_m], 'r--', label='理想线 y=x')
            plt.title(f'缺失数据 - 节点 {node_id_to_plot} - {feature} ({dataset_type})')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    # 模拟缺失数据对测试集的影响
    missing_nodes_list = [[4,8], [4,7,8,3],[6,0],[11,0,6,10],[4,7,8,3,9,1,2,5],[11,10,0,6,12,5,2,1],[4,7,8,3,9,1],[11,10,0,6,12,5],[4,7,8,3,9,1,2,5,12,11],[11,10,0,6,12,5,1,9,3,2],[4,7,8,3,9,1,2,5,12,11,10,0],[11,10,0,6,12,5,1,9,3,7,8,2]]
    for missing_nodes in missing_nodes_list:
        metrics_missing_test = simulate_missing_nodes_and_evaluate(model, X_test, y_test, scaler, missing_nodes)
        print(f'缺失节点 {missing_nodes} 时的测试集指标:')
        print(
            f"MAE: {metrics_missing_test['MAE']:.4f}, RMSE: {metrics_missing_test['RMSE']:.4f}, MAPE: {metrics_missing_test['MAPE']:.2f}%, R²: {metrics_missing_test['R²']:.4f}")

    # 模拟缺失数据对验证集的影响
    for missing_nodes in [[1,7,8,12], [0,3,4,9]]:
        metrics_missing_val = simulate_missing_nodes_and_evaluate(model, X_val, y_val, scaler, missing_nodes)
        print(f'缺失节点 {missing_nodes} 时的验证集指标:')
        print(
            f"MAE: {metrics_missing_val['MAE']:.4f}, RMSE: {metrics_missing_val['RMSE']:.4f}, MAPE: {metrics_missing_val['MAPE']:.2f}%, R²: {metrics_missing_val['R²']:.4f}")

    # 比较完整数据和缺失数据的评估指标
    print("\n完整数据的验证集指标:")
    print(f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")

    print("\n完整数据的测试集指标:")
    print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")

    # 可视化缺失数据情况下的预测结果变化
    for missing_nodes in [[8], [9]]:
        visualize_missing_impact(model, X_test, y_test, scaler, missing_nodes, node_id_to_plot, '测试集')
        visualize_missing_impact(model, X_val, y_val, scaler, missing_nodes, node_id_to_plot, '验证集')


if __name__ == '__main__':
    main()

