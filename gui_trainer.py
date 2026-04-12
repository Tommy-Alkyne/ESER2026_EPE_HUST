import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import os
import queue

class FluidTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("T型流体场 PINN 训练控制台")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        
        self.process = None
        self.output_queue = queue.Queue()
        
        self.create_widgets()
        self.check_queue()

    def create_widgets(self):
        # 整体布局: 左侧参数配置，右侧终端输出
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_paned)
        right_frame = ttk.Frame(main_paned)
        
        main_paned.add(left_frame, weight=1)
        main_paned.add(right_frame, weight=2)
        
        # ================= 左侧: 参数配置 =================
        # 1. 数据配置
        data_frame = ttk.LabelFrame(left_frame, text="数据配置 (Data Configuration)")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(data_frame, text="训练集 (Train File):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.train_file_var = tk.StringVar(value="p_v_256.hdf5")
        ttk.Entry(data_frame, textvariable=self.train_file_var, width=20).grid(row=0, column=1, padx=5)
        
        ttk.Label(data_frame, text="测试集 (Test File):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_file_var = tk.StringVar(value="p_v_256.hdf5")
        ttk.Entry(data_frame, textvariable=self.test_file_var, width=20).grid(row=1, column=1, padx=5)
        
        self.use_3d_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(data_frame, text="启用 3D 模型 (--ndim 3)", variable=self.use_3d_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # 2. 训练超参数
        hyper_frame = ttk.LabelFrame(left_frame, text="超参数配置 (Hyperparameters)")
        hyper_frame.pack(fill=tk.X, pady=5)
        
        params = [
            ("总轮数 (Epochs):", "epochs_var", "200"),
            ("批大小 (Batch Size):", "batch_size_var", "16"),
            ("学习率 (LR):", "lr_var", "0.003"),
            ("网络宽度 (Growth Rate):", "growth_rate_var", "16"),
            ("权重衰减 (Weight Decay):", "wd_var", "0.0005"),
        ]
        
        for i, (label_text, var_name, default_val) in enumerate(params):
            ttk.Label(hyper_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=default_val)
            setattr(self, var_name, var)
            ttk.Entry(hyper_frame, textvariable=var, width=15).grid(row=i, column=1, padx=5)

        # 3. 实验配置
        exp_frame = ttk.LabelFrame(left_frame, text="实验管理 (Experiment)")
        exp_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(exp_frame, text="实验组名称:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.exp_name_var = tk.StringVar(value="严格按照轮数")
        ttk.Entry(exp_frame, textvariable=self.exp_name_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(exp_frame, text="保存频率 (Ckpt Freq):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.ckpt_freq_var = tk.StringVar(value="50")
        ttk.Entry(exp_frame, textvariable=self.ckpt_freq_var, width=15).grid(row=1, column=1, padx=5)

        # 4. 控制按钮
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=15)
        
        self.start_btn = ttk.Button(control_frame, text="开始训练", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # ================= 左侧: GPU监控 =================
        gpu_frame = ttk.LabelFrame(left_frame, text="硬件监控 (System & GPU)")
        gpu_frame.pack(fill=tk.X, pady=5)
        
        self.gpu_util_var = tk.StringVar(value="GPU 使用率: 正在获取...")
        ttk.Label(gpu_frame, textvariable=self.gpu_util_var, font=("Consolas", 11, "bold"), foreground="blue").pack(anchor=tk.W, padx=5, pady=5)
        
        self.gpu_mem_var = tk.StringVar(value="显存使用量: 正在获取...")
        ttk.Label(gpu_frame, textvariable=self.gpu_mem_var, font=("Consolas", 11, "bold"), foreground="purple").pack(anchor=tk.W, padx=5, pady=5)

        self.update_gpu_status()

        # ================= 右侧: 终端输出 =================
        log_frame = ttk.LabelFrame(right_frame, text="实时训练日志 (Console Output)")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(log_frame, bg="black", fg="lightgreen", font=("Consolas", 12))
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_gpu_status(self):
        try:
            # 调用 nvidia-smi 提取单卡的 GPU 使用率、已用显存、总显存
            smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            lines = smi_output.strip().split('\n')
            if lines:
                util, mem_used, mem_total = lines[0].split(', ')
                self.gpu_util_var.set(f"GPU 使用率: {util: >3} %")
                self.gpu_mem_var.set(f"显存使用量: {mem_used: >5} / {mem_total} MB")
        except Exception:
            self.gpu_util_var.set("GPU监控: 未检测到NVIDIA GPU或驱动异常")
            self.gpu_mem_var.set("显存使用量: -- / -- MB")
            
        # 每 1.5 秒异步刷新一次状态
        self.root.after(1500, self.update_gpu_status)

    def start_training(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("警告", "训练任务已经在运行中！")
            return
            
        # 组装命令
        cmd = ["python", "train.py"]
        
        if self.train_file_var.get().strip():
            cmd.extend(["--train-file", self.train_file_var.get().strip()])
        if self.test_file_var.get().strip():
            cmd.extend(["--test-file", self.test_file_var.get().strip()])
        if self.use_3d_var.get():
            cmd.extend(["--ndim", "3"])
            
        cmd.extend([
            "--epochs", self.epochs_var.get().strip(),
            "--batch-size", self.batch_size_var.get().strip(),
            "--lr", self.lr_var.get().strip(),
            "--growth-rate", self.growth_rate_var.get().strip(),
            "--weight-decay", self.wd_var.get().strip(),
            "--exp-name", self.exp_name_var.get().strip(),
            "--ckpt-freq", self.ckpt_freq_var.get().strip(),
            "--plot-freq", self.ckpt_freq_var.get().strip() # 默认与保存频率一致
        ])
        
        self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, "启动命令: " + " ".join(cmd) + "\n" + "-"*60 + "\n")
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # 启动子进程
        try:
            # 兼容 Windows和Linux，通过PIPE获取输出
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, # 将 stderr 重定向到 stdout
                text=True, 
                bufsize=1, 
                universal_newlines=True
            )
            
            # 开启线程读取输出，防止阻塞 GUI
            threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
            
        except Exception as e:
            self.console.insert(tk.END, f"启动失败: {str(e)}\n")
            self.stop_training()

    def read_output(self, process):
        for line in process.stdout:
            self.output_queue.put(line)
            
        process.stdout.close()
        process.wait()
        self.output_queue.put(f"\n[{'='*20} 训练进程已结束 (Exit code: {process.returncode}) {'='*20}]\n")

    def check_queue(self):
        """定时检查队列，将内容更新到文本框（线程安全）"""
        while not self.output_queue.empty():
            line = self.output_queue.get()
            self.console.insert(tk.END, line)
            self.console.see(tk.END)
            
            # 如果捕捉到进程结束标记，恢复按钮状态
            if "训练进程已结束" in line:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)

        # 每 100 毫秒循环检查一次
        self.root.after(100, self.check_queue)

    def stop_training(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process = None
            self.console.insert(tk.END, "\n[由于用户操作，进程已被强制终止！]\n")
            
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    
    # 【提高清晰度】强制放大UI比例 (兼容Linux/Windows)
    try:
        root.tk.call('tk', 'scaling', 1.25)
    except Exception:
        pass

    # 尝试应用更现代的主题与全局大字体
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    style.configure('.', font=('Microsoft YaHei', 11))
    style.configure('TLabelframe.Label', font=('Microsoft YaHei', 11, 'bold'))
        
    app = FluidTrainerGUI(root)
    root.mainloop()