import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class ModelViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorFlow Model Viewer")
        self.model = None
        self.auxiliary_model = None
        self.input_data = None  # 儲存 CSV 檔案的路徑
        self.output_data = None
        self.create_widgets()

    def create_widgets(self):
        # Input Selection (改為 CSV 檔案)
        tk.Label(self.root, text="Input CSV File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Label(self.root, text="請先選取 csv 預測資料").grid(row=0, column=2, padx=0, pady=5, sticky="w")
        self.input_path_entry = tk.Entry(self.root, width=50)
        self.input_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self.root, text="Browse", command=self.browse_input).grid(row=0, column=3, padx=5, pady=5)
        
        # Main Model Selection (下拉式選單)
        tk.Label(self.root, text="Select Main Model:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(self.root)
        self.model_var.set("")  # 初始值設為空
        self.model_dropdown = ttk.Combobox(self.root, textvariable=self.model_var, values=[""], state="readonly") # 初始選項設為空字串的列表
        self.model_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Auxiliary Model Selection (下拉式選單)
        tk.Label(self.root, text="Select Auxiliary Model:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.auxiliary_model_var = tk.StringVar(self.root)
        self.auxiliary_model_var.set("")  # 初始值設為空
        self.auxiliary_model_dropdown = ttk.Combobox(self.root, textvariable=self.auxiliary_model_var, values=[""], state="readonly") # 初始選項設為空字串的列表
        self.auxiliary_model_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Run Button
        tk.Button(self.root, text="Run Inference", command=self.run_inference).grid(row=3, column=1, padx=5, pady=10)

        # Output Display
        tk.Label(self.root, text="Output:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.output_display = tk.Text(self.root, height=10, width=80)
        self.output_display.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

    def browse_input(self):
        filepath = filedialog.askopenfilename(title="Select Input CSV File",
                                             filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        self.input_path_entry.delete(0, tk.END)
        self.input_path_entry.insert(0, filepath)
        self.input_data = filepath  # 儲存 CSV 檔案路徑
        self.update_dropdown_options()

    def update_dropdown_options(self):
        # 讀取 CSV 檔案，動態判斷選單選項
        if self.input_data:
            try:
                df = pd.read_csv(self.input_data)
                # 根據 CSV 的內容，動態設定選單選項
                # 這裡假設你的 CSV 檔案至少包含 'PM10', 'PM2.5', 'aqi' 這幾個欄位
                if 'PM10' in df.columns and 'PM2.5' in df.columns and 'aqi' in df.columns:
                   self.model_dropdown.config(values=["RF LSTM","AE LSTM"])
                   self.auxiliary_model_dropdown.config(values=["Random Forest","Autoencoder"])

                else:
                    messagebox.showerror("Error", "CSV file must contain 'PM10', 'PM2.5', and 'aqi' columns.")
            except Exception as e:
                messagebox.showerror("Error", f"Error reading CSV file: {e}")

    def run_inference(self):
        selected_model = self.model_var.get()
        selected_auxiliary_model = self.auxiliary_model_var.get()

        if not self.input_data:
            messagebox.showerror("Error", "Please select an input CSV file.")
            return

        if selected_model == "RF LSTM" and selected_auxiliary_model == "Random Forest":
            self.run_rf_lstm_with_rf()
        elif selected_model == "AE LSTM":
            self.run_ae_lstm()
        else:
            messagebox.showerror("Error", "Invalid model and auxiliary model combination.")

    def run_rf_lstm_with_rf(self):
        try:
            df = pd.read_csv(self.input_data)
            data = df[['PM10', 'PM2.5','aqi']]
            loaded_LSTM_RANDOMFOREST = tf.keras.models.load_model(r"C:\Users\martin\Desktop\GUI\model用\LSTM_RANDOMFOREST_6_3_rmse59.keras")
            time_steps = 6
            test = data
            test_temp = test.to_numpy()
            ##scale and data preprocessing
            with open(r"C:\Users\martin\Desktop\GUI\model用\scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            with open(r"C:\Users\martin\Desktop\GUI\model用\scaler2.pkl", 'rb') as f:
                scaler2 = pickle.load(f)
            test_np= scaler.transform(test_temp)
            x_test = []
            y_test = []
            x_test.append(test_np[0:time_steps])
            y_test= data['aqi']#所有的aqi
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            ## train
            j = 0 
            num_steps_to_show = 3
            initial_sequence = []
            initial_sequence.append(x_test[0][0:time_steps])
            initial_sequence = np.array(initial_sequence)
            initial_sequence = np.array(initial_sequence)
            sequence = []
            temp = x_test.shape[2]

            for i in range(num_steps_to_show):
                new_prediction = loaded_LSTM_RANDOMFOREST.predict(initial_sequence)
            
                initial_sequence = initial_sequence[0][1:]
                
                initial_sequence = np.append(initial_sequence,new_prediction,axis=0).reshape(-1,time_steps,x_test.shape[2])
                
                sequence.append(new_prediction[0][temp-1])

            sequence = scaler2.inverse_transform(np.array(sequence).reshape(3,1))
            sequence=sequence.astype(int)
            
            self.display_output(sequence)
            
            ###
            plt.figure(figsize=(8, 4))
            plt.plot(sequence[:num_steps_to_show], label='Predicted Values', color='orange', linestyle='-')
            plt.xlabel('Index')
            plt.ylabel('aqi')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error during inference: {e}")

    def run_ae_lstm(self):
        try:
            df = pd.read_csv(self.input_data)
            X = df.drop(columns=['time','aqi','WD_HR','WIND_SPEED'])
            # 載入 Autoencoder 前需要之scaler
            scaler_filename=r"C:\Users\martin\Desktop\GUI\model用\scaler_AE_test.pkl"
            with open(scaler_filename, 'rb') as file:
               loaded_scaler = pickle.load(file)

            # 使用載入的 scaler 進行標準化
            X = loaded_scaler.transform(X)

            ##載入AE
            loaded_encoder = load_model(r"C:\Users\martin\Desktop\GUI\model用\encoder_model_v2.keras")
            X_encoded_loaded = loaded_encoder.predict(X)

            # 將壓縮特徵轉為 DataFrame
            X_encoded_df = pd.DataFrame(X_encoded_loaded, columns=[f'encoded_{i}' for i in range(X_encoded_loaded.shape[1])])

            # 合併壓縮特徵與目標變數
            data_with_encoded = pd.concat([X_encoded_df, df[['aqi']].reset_index(drop=True)], axis=1)

            ##載入AE_LSTM model
            loaded_AE_LSTM = tf.keras.models.load_model(r"C:\Users\martin\Desktop\GUI\model用\LSTM_AE_V2_6_3_rmse_43.keras")
            time_steps = 6
            test =  data_with_encoded
            test_temp = test.to_numpy()
            ##before lstm scale and data preprocessing
            with open(r"C:\Users\martin\Desktop\GUI\model用\scaler_MINMAX_AE_1.pkl", 'rb') as f:
               scaler = pickle.load(f)
            with open(r"C:\Users\martin\Desktop\GUI\model用\scaler_MINMAX_inverse_AE_2.pkl", 'rb') as f:
               scaler2 = pickle.load(f)
            test_np= scaler.transform(test_temp)
            x_test = []
            y_test = []
            x_test.append(test_np[0:time_steps])
            y_test= df['aqi']#所有的aqi
            x_test = np.array(x_test)
            y_test = np.array(y_test)


            ## predict
            j = 0 
            num_steps_to_show = 3
            initial_sequence = []
            initial_sequence.append(x_test[0][0:time_steps])
            initial_sequence = np.array(initial_sequence)
            initial_sequence = np.array(initial_sequence)
            sequence = []
            temp = x_test.shape[2]

            for i in range(num_steps_to_show):
               new_prediction = loaded_AE_LSTM.predict(initial_sequence)
            
               initial_sequence = initial_sequence[0][1:]
                
               initial_sequence = np.append(initial_sequence,new_prediction,axis=0).reshape(-1,time_steps,x_test.shape[2])
                
               sequence.append(new_prediction[0][temp-1])

            sequence = scaler2.inverse_transform(np.array(sequence).reshape(3,1))
            sequence=sequence.astype(int)
            
            self.display_output(sequence)
            
            ### 畫圖
            plt.figure(figsize=(8, 4))
            plt.plot(sequence[:num_steps_to_show], label='Predicted Values', color='orange', linestyle='-')
            plt.xlabel('Index')
            plt.ylabel('aqi')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during inference: {e}")
    
    def display_output(self, sequence):
        self.output_display.delete(1.0, tk.END)  # 清空舊的輸出

        for i, value in enumerate(sequence):
            aqi_value = value[0]
            self.output_display.insert(tk.END, f"第{i+1}小時AQI預測值: {aqi_value}\n")

            if 0 <= aqi_value <= 50:
                self.output_display.insert(tk.END, "AQI 指標: 良好\n")
                self.output_display.insert(tk.END, "人體健康影響: 空氣品質為良好，污染程度低或無污染。\n")
                self.output_display.insert(tk.END, "一般民眾活動建議: 正常戶外活動\n")
                self.output_display.insert(tk.END, "敏感性族群活動建議: 正常戶外活動\n")
            elif 51 <= aqi_value <= 100:
                self.output_display.insert(tk.END, "AQI 指標: 普通\n")
                self.output_display.insert(tk.END, "人體健康影響: 空氣品質普通；但對非常少數之極敏感族群產生輕微影響。\n")
                self.output_display.insert(tk.END, "一般民眾活動建議: 正常戶外活動\n")
                self.output_display.insert(tk.END, "敏感性族群活動建議: 極特殊敏感族群建議注意可能產生的咳嗽或呼吸急促症狀，但仍可正常戶外活動。\n")
            elif 101 <= aqi_value <= 150:
                 self.output_display.insert(tk.END, "AQI 指標: 普通\n")
                 self.output_display.insert(tk.END, "人體健康影響: 空氣品質普通；但對非常少數之極敏感族群產生輕微影響。\n")
                 self.output_display.insert(tk.END, "一般民眾活動建議: 正常戶外活動\n")
                 self.output_display.insert(tk.END, "敏感性族群活動建議: 極特殊敏感族群建議注意可能產生的咳嗽或呼吸急促症狀，但仍可正常戶外活動。\n")
            elif aqi_value > 150:
                self.output_display.insert(tk.END, "AQI 指標: 對所有族群不健康\n")
                self.output_display.insert(tk.END, "人體健康影響: 對所有人的健康開始產生影響\n")
                self.output_display.insert(tk.END, "一般民眾活動建議: 應減少體力消耗，特別是減少戶外活動。\n")
                self.output_display.insert(tk.END, "敏感性族群活動建議: 建議留在室內並減少體力消耗活動，必要外出應配戴口罩。\n")
            self.output_display.insert(tk.END, "\n")  # 每筆資料間隔一行

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelViewerApp(root)
    root.mainloop()