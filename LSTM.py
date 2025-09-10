import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from matplotlib import font_manager



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保在使用CUDNN时，操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题



def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))  # 读取前100KB
    return result['encoding']


def load_data(water_quality_path, adjacency_path):
    if not os.path.exists(water_quality_path):
        raise FileNotFoundError(f"水质数据文件未找到: {water_quality_path}")
    if not os.path.exists(adjacency_path):
        raise FileNotFoundError(f"邻接矩阵文件未找到: {adjacency_path}")


    water_quality_encoding = detect_encoding(water_quality_path)
    adjacency_encoding = detect_encoding(adjacency_path)

    print(f"水质数据编码: {water_quality_encoding}")
    print(f"邻接矩阵编码: {adjacency_encoding}")

    water_quality_df = pd.read_csv(water_quality_path, encoding=water_quality_encoding, parse_dates=['监测时间'])
    adjacency_df = pd.read_csv(adjacency_path, header=None, encoding=adjacency_encoding)

    return water_quality_df, adjacency_df


def preprocess_data(water_quality_df):
    scaler = StandardScaler()

    feature_cols = ['溶解氧', '高锰酸盐指数', '氨氮', '总磷', '总氮']  
    features = water_quality_df[feature_cols].values
    scaled_features = scaler.fit_transform(features)
    water_quality_df[feature_cols] = scaled_features
    return water_quality_df, scaler, feature_cols



def create_sequences_all_nodes(df, seq_length, feature_cols):
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
            features = time_data[feature_cols].values  
            if features.shape[0] != num_nodes:
                raise ValueError(f"时间戳 {timestamps[t]} 的节点数量不匹配。")
            seq.append(features)
        target_time = i + seq_length
        target_data = df[df['监测时间'] == timestamps[target_time]].sort_values('node_id')[feature_cols].values
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


class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_lstm, num_nodes, dropout=0.5, num_lstm_layers=1):
        super(LSTMModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_lstm = hidden_lstm
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout

        
        self.lstm = nn.LSTM(input_size=num_nodes * num_features,
                            hidden_size=hidden_lstm,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0)  

       
        self.fc = nn.Linear(hidden_lstm, num_nodes * num_features)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_length, num_nodes, num_features)
        """
        batch_size, seq_length, num_nodes, num_features = x.size()
        
        x = x.view(batch_size, seq_length, num_nodes * num_features)
        
        lstm_out, _ = self.lstm(x)  
        lstm_out = lstm_out[:, -1, :]  
        
        out = self.fc(lstm_out) 
        
        out = out.view(batch_size, num_nodes, num_features)
        return out



def main():
  
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    
    water_quality_path = r'D:.csv'  
    adjacency_path = r'.csv'  

   
    water_quality_df, adjacency_df = load_data(water_quality_path, adjacency_path)
    print("数据加载完成。")

    
   
    water_quality_df = water_quality_df.ffill().bfill()

    water_quality_df, scaler, feature_cols = preprocess_data(water_quality_df)
    print("数据归一化完成。")

    
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler 已保存为 'scaler.pkl'。")

  
    target_feature_cols = ['总磷', '总氮']  
    target_feature_indices = [feature_cols.index(col) for col in target_feature_cols]

   
    num_nodes = water_quality_df['node_id'].nunique()
    num_features = len(feature_cols)  
    print(f"节点数: {num_nodes}, 特征数: {num_features}")

    
    SEQ_LENGTH = 10  
    X, y = create_sequences_all_nodes(water_quality_df, SEQ_LENGTH, feature_cols)
    print(f"所有节点的序列创建完成。X shape: {X.shape}, y shape: {y.shape}")

   
   
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
  
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)
    print(f"训练集、验证集和测试集划分完成。")
    print(f"训练集样本数: {X_train.shape[0]}, 验证集样本数: {X_val.shape[0]}, 测试集样本数: {X_test.shape[0]}")

    
    train_dataset = WaterQualityDataset(X_train, y_train)
    val_dataset = WaterQualityDataset(X_val, y_val)
    test_dataset = WaterQualityDataset(X_test, y_test)  
    BATCH_SIZE = 64

    
    def get_dataloader(dataset, batch_size, shuffle=False, seed=42):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    train_loader = get_dataloader(train_dataset, BATCH_SIZE, shuffle=True, seed=42)
    val_loader = get_dataloader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_dataset, BATCH_SIZE, shuffle=False)  
    print("数据加载器创建完成。")

 
    HIDDEN_LSTM = 128
    DROPOUT = 0.3
    NUM_LSTM_LAYERS = 1
    model = LSTMModel(num_features=num_features, hidden_lstm=HIDDEN_LSTM,
                      num_nodes=num_nodes, dropout=DROPOUT, num_lstm_layers=NUM_LSTM_LAYERS)
    model = model.to(device)
    print("模型初始化完成。")

 
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
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
        """
       
        """
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

       
        y_true_target = y_true[:, target_feature_indices]
        y_pred_target = y_pred[:, target_feature_indices]

        mae = mean_absolute_error(y_true_target, y_pred_target)
        mse = mean_squared_error(y_true_target, y_pred_target)
        rmse = np.sqrt(mse)
       
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true_target - y_pred_target) / (y_true_target + epsilon))) * 100
        r2 = r2_score(y_true_target, y_pred_target)
        return mae, rmse, mape, r2

    
    def simulate_missing_nodes(X, missing_nodes):
        """
        模拟输入数据中缺失某些节点的特征值。
        参数:
        - X: numpy array of shape (num_samples, seq_length, num_nodes, num_features)
        - missing_nodes: list of int, 要模拟缺失的节点编号
        返回:
        - X_missing: numpy array with missing nodes' features set to zero
        """
        X_missing = X.copy()
        X_missing[:, :, missing_nodes, :] = 0  
        return X_missing

   
    EPOCHS = 1000
    best_val_loss = float('inf')
    patience = 50
    trigger_times = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f'Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

       
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
    print(f'测试损失: {final_test_loss:.4f}')

    
    train_mae, train_rmse, train_mape, train_r2 = evaluate_metrics(model, train_loader, scaler)
    val_mae, val_rmse, val_mape, val_r2 = evaluate_metrics(model, val_loader, scaler)
    test_mae, test_rmse, test_mape, test_r2 = evaluate_metrics(model, test_loader, scaler)  
    print(
        f'Training MAE (总磷 & 总氮): {train_mae:.4f}, Training RMSE: {train_rmse:.4f}, Training MAPE: {train_mape:.2f}%, Training R²: {train_r2:.4f}')
    print(
        f'Validation MAE (总磷 & 总氮): {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}, Validation MAPE: {val_mape:.2f}%, Validation R²: {val_r2:.4f}')
    print(
        f'Test MAE (总磷 & 总氮): {test_mae:.4f}, Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.2f}%, Test R²: {test_r2:.4f}')  

    
    def inverse_transform_predictions(scaler, predictions):
        """
        scaler: StandardScaler 对象
        predictions: numpy array of shape (num_nodes, num_features)
        返回: 逆标准化后的预测值，形状为 (num_nodes, num_features)
        """
        
        original_scale = scaler.inverse_transform(predictions)
        
        df = pd.DataFrame(original_scale, columns=feature_cols)
        return df

    def predict(model, input_sequence, scaler):
        """
        input_sequence: numpy array of shape (seq_length, num_nodes, num_features)
        返回: 逆标准化后的 DataFrame，形状为 (num_nodes, num_features)
        """
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(input_sequence, dtype=torch.float).unsqueeze(0).to(
                device)  # (1, seq_length, num_nodes, num_features)
            prediction = model(input_tensor)  # (1, num_nodes, num_features)
            
            prediction = prediction.detach().cpu().numpy().reshape(num_nodes, num_features)
       
        prediction_original = inverse_transform_predictions(scaler, prediction)
        
        return prediction_original[target_feature_cols]

   
    sample_X = X_val[0]  # (seq_length, num_nodes, num_features)
    predicted_water_quality = predict(model, sample_X, scaler)
    print("预测结果（第一个验证样本，逆标准化后，仅包含总磷和总氮）:")
    print(predicted_water_quality)

    

    def visualize_predictions_scatter(y_true_val, y_pred_val, feature_names, node_id):
        """
        y_true_val: numpy array of shape (num_samples, num_features)
        y_pred_val: numpy array of shape (num_samples, num_features)
        feature_names: list of feature names
        node_id: int, the node to plot (0-12)
        """
        for feature_idx, feature in enumerate(feature_names):
            plt.figure(figsize=(6, 6))  
            plt.scatter(y_true_val[:, feature_idx], y_pred_val[:, feature_idx], alpha=0.5, label='Predicted vs Actual')
            max_val = max(y_true_val[:, feature_idx].max(), y_pred_val[:, feature_idx].max())
            min_val = min(y_true_val[:, feature_idx].min(), y_pred_val[:, feature_idx].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Line y=x')
            plt.title(f'Node {node_id} - {feature}: Predicted vs Actual Values')  
            plt.xlabel('Actual Value')  
            plt.ylabel('Predicted Value')  
            plt.legend()  
            plt.grid(True)  
            plt.show()

  
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_X)
           
            output = output.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
            batch_y = batch_y.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
        
            for i in range(batch_y.shape[0]):
                y_true_full = scaler.inverse_transform(batch_y[i])
                y_pred_full = scaler.inverse_transform(output[i])
                y_true_val.append(y_true_full[target_feature_indices])
                y_pred_val.append(y_pred_full[target_feature_indices])
    y_true_val = np.array(y_true_val).reshape(-1, len(target_feature_cols))
    y_pred_val = np.array(y_pred_val).reshape(-1, len(target_feature_cols))

    
    feature_names = target_feature_cols  
    node_id_to_plot = 0  

    
    if node_id_to_plot < 0 or node_id_to_plot >= num_nodes:
        raise ValueError(f"node_id_to_plot ({node_id_to_plot}) 超出范围 (0-{num_nodes - 1})")

    for feature_idx, feature in enumerate(feature_names):
        plt.figure(figsize=(6, 6)) 
        plt.scatter(y_true_val[:, feature_idx], y_pred_val[:, feature_idx], alpha=0.5, label='Predicted vs Actual')
        max_val = max(y_true_val[:, feature_idx].max(), y_pred_val[:, feature_idx].max())
        min_val = min(y_true_val[:, feature_idx].min(), y_pred_val[:, feature_idx].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Line y=x')
        plt.title(f'Node {node_id_to_plot} - {feature}: Predicted vs Actual Values')  
        plt.xlabel('Actual Value')  
        plt.ylabel('Predicted Value')  
        plt.legend()  
        plt.grid(True) 
        plt.show()

   
    def simulate_missing_nodes_and_evaluate(model, dataloader, scaler, missing_nodes):
        """
       
        """
     
        X_original = []
        y_original = []
        for batch_X, batch_y in dataloader:
            X_original.append(batch_X.numpy())
            y_original.append(batch_y.numpy())
        X_original = np.concatenate(X_original, axis=0)
        y_original = np.concatenate(y_original, axis=0)

        # 模拟缺失数据
        X_missing = simulate_missing_nodes(X_original, missing_nodes)

        # 创建数据集和数据加载器
        missing_dataset = WaterQualityDataset(X_missing, y_original)
        missing_loader = DataLoader(missing_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 评估指标
        mae, rmse, mape, r2 = evaluate_metrics(model, missing_loader, scaler)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}

    missing_nodes_1 = [0,1,2]
    metrics_missing_val_1 = simulate_missing_nodes_and_evaluate(model, val_loader, scaler, missing_nodes_1)
    print(f'验证集缺失节点 {missing_nodes_1} 时的指标:')
    print(
        f"MAE: {metrics_missing_val_1['MAE']:.4f}, RMSE: {metrics_missing_val_1['RMSE']:.4f}, MAPE: {metrics_missing_val_1['MAPE']:.2f}%, R²: {metrics_missing_val_1['R²']:.4f}")

    # 示例：缺失多个点位的数据
    missing_nodes_2 =  [6,0,10,11,12,5,2,1,9,3,7,8]
    metrics_missing_val_2 = simulate_missing_nodes_and_evaluate(model, val_loader, scaler, missing_nodes_2)
    print(f'验证集缺失节点 {missing_nodes_2} 时的指标:')
    print(
        f"MAE: {metrics_missing_val_2['MAE']:.4f}, RMSE: {metrics_missing_val_2['RMSE']:.4f}, MAPE: {metrics_missing_val_2['MAPE']:.2f}%, R²: {metrics_missing_val_2['R²']:.4f}")

    # 比较完整数据和缺失数据的验证集指标
    print("\n完整数据的验证集指标:")
    print(f"MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%, R²: {val_r2:.4f}")

    # 现在，在测试集上进行类似的缺失节点评估
    # 示例：缺失0号点位的数据
    metrics_missing_test_1 = simulate_missing_nodes_and_evaluate(model, test_loader, scaler, missing_nodes_1)
    print(f'\n测试集缺失节点 {missing_nodes_1} 时的指标:')
    print(
        f"MAE: {metrics_missing_test_1['MAE']:.4f}, RMSE: {metrics_missing_test_1['RMSE']:.4f}, MAPE: {metrics_missing_test_1['MAPE']:.2f}%, R²: {metrics_missing_test_1['R²']:.4f}")

    # 示例：缺失多个点位的数据
    metrics_missing_test_2 = simulate_missing_nodes_and_evaluate(model, test_loader, scaler, missing_nodes_2)
    print(f'测试集缺失节点 {missing_nodes_2} 时的指标:')
    print(
        f"MAE: {metrics_missing_test_2['MAE']:.4f}, RMSE: {metrics_missing_test_2['RMSE']:.4f}, MAPE: {metrics_missing_test_2['MAPE']:.2f}%, R²: {metrics_missing_test_2['R²']:.4f}")

    # 比较完整数据和缺失数据的测试集指标
    print("\n完整数据的测试集指标:")
    print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.4f}")

    # 可视化缺失数据情况下的预测结果变化
    def visualize_missing_impact(model, original_loader, missing_nodes, scaler, node_id_to_plot):
        """
        可视化缺失数据情况下的预测结果变化。
        参数:
        - model: 训练好的模型
        - original_loader: 原始数据的 DataLoader
        - missing_nodes: list of int, 要模拟缺失的节点编号
        - scaler: 数据标准化的 scaler
        - node_id_to_plot: int, 要绘制的节点编号
        """
        # 获取原始数据
        y_true_full = []
        y_pred_full = []
        with torch.no_grad():
            for batch_X, batch_y in original_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                output = output.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                batch_y = batch_y.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                for i in range(batch_y.shape[0]):
                    y_true_full_transformed = scaler.inverse_transform(batch_y[i])
                    y_pred_full_transformed = scaler.inverse_transform(output[i])
                    y_true_full.append(y_true_full_transformed[target_feature_indices])
                    y_pred_full.append(y_pred_full_transformed[target_feature_indices])
        y_true_full = np.array(y_true_full).reshape(-1, len(target_feature_cols))
        y_pred_full = np.array(y_pred_full).reshape(-1, len(target_feature_cols))

        # 模拟缺失数据
        X_original = []
        y_original = []
        for batch_X, batch_y in original_loader:
            X_original.append(batch_X.numpy())
            y_original.append(batch_y.numpy())
        X_original = np.concatenate(X_original, axis=0)
        y_original = np.concatenate(y_original, axis=0)
        X_missing = simulate_missing_nodes(X_original, missing_nodes)

        # 创建缺失数据的 DataLoader
        missing_dataset = WaterQualityDataset(X_missing, y_original)
        missing_loader = DataLoader(missing_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 获取缺失数据的所有真实值和预测值
        y_true_missing = []
        y_pred_missing = []
        with torch.no_grad():
            for batch_X, batch_y in missing_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                output = output.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                batch_y = batch_y.detach().cpu().numpy().reshape(batch_y.size(0), num_nodes, num_features)
                for i in range(batch_y.shape[0]):
                    y_true_missing_transformed = scaler.inverse_transform(batch_y[i])
                    y_pred_missing_transformed = scaler.inverse_transform(output[i])
                    y_true_missing.append(y_true_missing_transformed[target_feature_indices])
                    y_pred_missing.append(y_pred_missing_transformed[target_feature_indices])
        y_true_missing = np.array(y_true_missing).reshape(-1, len(target_feature_cols))
        y_pred_missing = np.array(y_pred_missing).reshape(-1, len(target_feature_cols))

        # 绘制指定节点的每个目标特征的实际值与预测值的散点图

    for feature_idx, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_true_full[:, feature_idx], y_pred_full[:, feature_idx], alpha=0.3, label='完整数据')
        max_val = max(y_true_full[:, feature_idx].max(), y_pred_full[:, feature_idx].max())
        min_val = min(y_true_full[:, feature_idx].min(), y_pred_full[:, feature_idx].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 y=x')
        plt.title(f'完整数据 - 节点 {node_id_to_plot} - {feature}')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True)



    # 可视化缺失0号点位的数据影响（验证集）
    visualize_missing_impact(model, val_loader, missing_nodes_1, scaler, node_id_to_plot)

    # 可视化缺失多个点位的数据影响（验证集）
    visualize_missing_impact(model, val_loader, missing_nodes_2, scaler, node_id_to_plot)

    # 可视化缺失0号点位的数据影响（测试集）
    visualize_missing_impact(model, test_loader, missing_nodes_1, scaler, node_id_to_plot)

    # 可视化缺失多个点位的数据影响（测试集）
    visualize_missing_impact(model, test_loader, missing_nodes_2, scaler, node_id_to_plot)




if __name__ == '__main__':
    main()