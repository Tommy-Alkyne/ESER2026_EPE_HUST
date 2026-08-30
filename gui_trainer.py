"""T 型流体场 PINN 训练控制台（Tkinter）。

修复内容
--------
* 用 `sys.executable` 启动子进程，不再写死 "python"（venv/conda 下会调错解释器）。
* 默认实验名改成 ASCII，避免中文路径在跨平台/跨工具链下出问题。
* 暴露新增的关键参数：val_split、channel_order、物理项权重、domain_size、
  rho/nu、掩码开关 —— 这些以前是硬编码在 train.py 里的，改一次要动源码。
* `--ndim 3` 默认关闭（旧版默认勾选 3D，多数人其实在跑 2D）。
* 停止训练改为先 terminate 再等待，超时才 kill，避免留下僵尸进程。
* 子进程以本文件所在目录为工作目录，从任意路径启动 GUI 都能跑。
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


class FluidTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('T 型流体场 PINN 训练控制台')
        self.root.geometry('1180x820')
        self.root.minsize(1040, 700)

        self.process = None
        self.output_queue = queue.Queue()

        self.create_widgets()
        self.check_queue()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    # ------------------------------------------------------------------ UI
    def create_widgets(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        canvas = tk.Canvas(left, highlightthickness=0, width=430)
        scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=form, anchor='nw')
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.vars = {}

        # ---------------------------------------------------------- 数据
        data = ttk.LabelFrame(form, text='数据配置')
        data.pack(fill=tk.X, pady=4, padx=4)
        self._file_row(data, 0, '训练集:', 'train_file', 'dataset/p_v_256.hdf5')
        self._file_row(data, 1, '测试集(可空):', 'test_file', '')
        self._entry_row(data, 2, '验证集比例 val-split:', 'val_split', '0.2')
        self.use_3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data, text='3D 体素模型 (--ndim 3)',
                        variable=self.use_3d_var).grid(
            row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=4)
        ttk.Label(data, text='测试集留空时，会从训练集里按比例切出互不重叠的验证集',
                  foreground='gray').grid(row=4, column=0, columnspan=3,
                                          sticky=tk.W, padx=5)

        # ---------------------------------------------------------- 物理
        phys = ttk.LabelFrame(form, text='物理设置（PINN）')
        phys.pack(fill=tk.X, pady=4, padx=4)
        ttk.Label(phys, text='通道顺序:').grid(row=0, column=0, sticky=tk.W,
                                           padx=5, pady=4)
        self.channel_order_var = tk.StringVar(value='pvv')
        ttk.Combobox(phys, textvariable=self.channel_order_var, width=22,
                     state='readonly',
                     values=['pvv', 'vvp']).grid(row=0, column=1, padx=5)
        ttk.Label(phys, text='pvv=(P,Vx,Vy)  vvp=(Vx,Vy,P)',
                  foreground='red').grid(row=1, column=0, columnspan=3,
                                         sticky=tk.W, padx=5)
        self._entry_row(phys, 2, '计算域尺寸 Ly Lx:', 'domain_size', '1.0 1.0')
        self._entry_row(phys, 3, '密度 rho:', 'rho', '1.0')
        self._entry_row(phys, 4, '运动粘度 nu:', 'nu', '0.01')
        self._entry_row(phys, 5, '散度权重:', 'div_weight', '0.05')
        self._entry_row(phys, 6, '动量权重:', 'ns_weight', '0.005')
        self._entry_row(phys, 7, '平滑权重:', 'smooth_weight', '0.005')
        self._entry_row(phys, 8, '物理项预热比例:', 'physics_warmup', '0.1')
        self.no_mask_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(phys, text='关闭 T 型掩码（在整幅网格上训练）',
                        variable=self.no_mask_var).grid(
            row=9, column=0, columnspan=3, sticky=tk.W, padx=5, pady=4)

        # ---------------------------------------------------------- 超参
        hyper = ttk.LabelFrame(form, text='训练超参数')
        hyper.pack(fill=tk.X, pady=4, padx=4)
        for i, (label, key, default) in enumerate([
                ('总轮数 epochs:', 'epochs', '200'),
                ('批大小 batch-size:', 'batch_size', '16'),
                ('学习率 lr:', 'lr', '0.003'),
                ('网络宽度 growth-rate:', 'growth_rate', '16'),
                ('初始特征 init-features:', 'init_features', '48'),
                ('权重衰减 weight-decay:', 'weight_decay', '0.0005'),
                ('梯度裁剪 grad-clip:', 'grad_clip', '1.0'),
                ('随机种子 seed:', 'seed', '1')]):
            self._entry_row(hyper, i, label, key, default)

        # ---------------------------------------------------------- 实验
        exp = ttk.LabelFrame(form, text='实验管理')
        exp.pack(fill=tk.X, pady=4, padx=4)
        self._entry_row(exp, 0, '实验名:', 'exp_name', 'run01')
        self._entry_row(exp, 1, '存档频率 ckpt-freq:', 'ckpt_freq', '50')
        self._entry_row(exp, 2, '出图频率 plot-freq:', 'plot_freq', '50')

        # ---------------------------------------------------------- 控制
        control = ttk.Frame(form)
        control.pack(fill=tk.X, pady=10, padx=4)
        self.start_btn = ttk.Button(control, text='开始训练',
                                    command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
        self.stop_btn = ttk.Button(control, text='停止训练',
                                   command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=3)
        ttk.Button(control, text='运行诊断',
                   command=self.run_diagnose).pack(side=tk.LEFT, expand=True,
                                                   fill=tk.X, padx=3)

        # ---------------------------------------------------------- 监控
        gpu = ttk.LabelFrame(form, text='硬件监控')
        gpu.pack(fill=tk.X, pady=4, padx=4)
        self.gpu_util_var = tk.StringVar(value='GPU 使用率: 获取中...')
        ttk.Label(gpu, textvariable=self.gpu_util_var,
                  font=('Consolas', 10, 'bold'), foreground='blue').pack(
            anchor=tk.W, padx=5, pady=3)
        self.gpu_mem_var = tk.StringVar(value='显存: 获取中...')
        ttk.Label(gpu, textvariable=self.gpu_mem_var,
                  font=('Consolas', 10, 'bold'), foreground='purple').pack(
            anchor=tk.W, padx=5, pady=3)
        self.update_gpu_status()

        # ---------------------------------------------------------- 日志
        log = ttk.LabelFrame(right, text='实时训练日志')
        log.pack(fill=tk.BOTH, expand=True, pady=5)
        self.console = scrolledtext.ScrolledText(
            log, bg='#101010', fg='#9CDCFE', font=('Consolas', 11),
            insertbackground='white')
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _entry_row(self, parent, row, label, key, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W,
                                           padx=5, pady=3)
        var = tk.StringVar(value=default)
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=24).grid(row=row, column=1,
                                                           padx=5, sticky=tk.W)

    def _file_row(self, parent, row, label, key, default):
        self._entry_row(parent, row, label, key, default)
        ttk.Button(parent, text='...', width=3,
                   command=lambda k=key: self._browse(k)).grid(row=row, column=2,
                                                               padx=2)

    def _browse(self, key):
        path = filedialog.askopenfilename(
            initialdir=os.path.join(PROJECT_DIR, 'dataset'),
            filetypes=[('HDF5', '*.hdf5 *.h5'), ('全部文件', '*.*')])
        if path:
            try:
                path = os.path.relpath(path, PROJECT_DIR)
            except ValueError:
                pass
            self.vars[key].set(path)

    # -------------------------------------------------------------- GPU
    def update_gpu_status(self):
        try:
            out = subprocess.check_output(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL)
            util, used, total = out.strip().split('\n')[0].split(', ')
            self.gpu_util_var.set('GPU 使用率: {:>3} %'.format(util))
            self.gpu_mem_var.set('显存: {:>6} / {} MB'.format(used, total))
        except Exception:
            self.gpu_util_var.set('GPU: 未检测到 NVIDIA 设备（将使用 CPU）')
            self.gpu_mem_var.set('显存: -- / -- MB')
        self.root.after(2000, self.update_gpu_status)

    # -------------------------------------------------------------- 训练
    def build_command(self):
        v = {k: var.get().strip() for k, var in self.vars.items()}
        cmd = [sys.executable, '-u', 'train.py']

        if v['train_file']:
            cmd += ['--train-file', v['train_file']]
        if v['test_file']:
            cmd += ['--test-file', v['test_file']]
        if self.use_3d_var.get():
            cmd += ['--ndim', '3']
        if self.no_mask_var.get():
            cmd += ['--no-mask']

        cmd += ['--channel-order', self.channel_order_var.get()]
        if v['domain_size']:
            cmd += ['--domain-size'] + v['domain_size'].replace(',', ' ').split()

        simple = [('val_split', '--val-split'), ('rho', '--rho'), ('nu', '--nu'),
                  ('div_weight', '--div-weight'), ('ns_weight', '--ns-weight'),
                  ('smooth_weight', '--smooth-weight'),
                  ('physics_warmup', '--physics-warmup'),
                  ('epochs', '--epochs'), ('batch_size', '--batch-size'),
                  ('lr', '--lr'), ('growth_rate', '--growth-rate'),
                  ('init_features', '--init-features'),
                  ('weight_decay', '--weight-decay'),
                  ('grad_clip', '--grad-clip'), ('seed', '--seed'),
                  ('exp_name', '--exp-name'), ('ckpt_freq', '--ckpt-freq'),
                  ('plot_freq', '--plot-freq')]
        for key, flag in simple:
            if v.get(key):
                cmd += [flag, v[key]]
        return cmd

    def start_training(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning('警告', '训练任务已经在运行中')
            return
        try:
            cmd = self.build_command()
        except Exception as exc:
            messagebox.showerror('参数错误', str(exc))
            return

        self.console.delete(1.0, tk.END)
        self.console.insert(tk.END, '工作目录: {}\n启动命令: {}\n{}\n'
                            .format(PROJECT_DIR, ' '.join(cmd), '-' * 68))
        self._launch(cmd)

    def run_diagnose(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning('警告', '请先等训练进程结束')
            return
        self._launch([sys.executable, '-u', 'diagnose.py'])

    def _launch(self, cmd):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        try:
            self.process = subprocess.Popen(
                cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                encoding='utf-8', errors='replace', env=env)
        except Exception as exc:
            self.console.insert(tk.END, '启动失败: {}\n'.format(exc))
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
        threading.Thread(target=self.read_output, args=(self.process,),
                         daemon=True).start()

    def read_output(self, process):
        try:
            for line in process.stdout:
                self.output_queue.put(line)
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass
            process.wait()
            self.output_queue.put(
                '\n[{} 进程结束 (exit code: {}) {}]\n'
                .format('=' * 22, process.returncode, '=' * 22))

    def check_queue(self):
        while not self.output_queue.empty():
            line = self.output_queue.get()
            self.console.insert(tk.END, line)
            self.console.see(tk.END)
            if '进程结束' in line:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
        self.root.after(100, self.check_queue)

    def stop_training(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            self.console.insert(tk.END, '\n[正在请求进程退出...]\n')
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.console.insert(tk.END, '[进程未响应，强制结束]\n')
                proc.kill()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askokcancel('退出', '训练仍在运行，确定要退出吗？'):
                return
            self.stop_training()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.25)
    except Exception:
        pass
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    for family in ('Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC'):
        try:
            style.configure('.', font=(family, 10))
            style.configure('TLabelframe.Label', font=(family, 10, 'bold'))
            break
        except Exception:
            continue
    FluidTrainerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
