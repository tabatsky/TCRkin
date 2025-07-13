import csv
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numpy import exp, log10, abs
from numpy.linalg import eig
import scipy
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.optimize import minimize, direct, rosen_der, rosen, shgo
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import CubicSpline
import threading
from numdifftools import Hessian, Jacobian
import matplotlib
matplotlib.use('Qt5Agg')

# Constants
DEFAULT_FILE = '20220824_test17.csv'
# DEFAULT_FILE = '20220609_N4.csv'
DEBUG = False
DIRECT_ITER = 100000
DIRECT_LOCALLY_BIASED = False
DIRECT_EPSILON = 1e-3
L1 = 10.0
L2 = 5.0
L3 = 2.0
BOUNDS = [(2, 5), (log10(3), log10(300)),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1),
          (-5, 1), (-5, 1), (-5, 1)]

IS_ING = [False, True]
LETTERS = ['A', 'B', 'C', 'D']
PARAM_NAMES = ['A', 'B']
for is_ing in IS_ING:
    is_ing_str = '2' if is_ing else '1'
    for letter in LETTERS:
        for pn in ['kon', 'koff', 'krel']:
            PARAM_NAMES.append(pn + '_' + letter + is_ing_str)


class KineticModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kinetic Model Fitting Tool")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        print(screen_width, screen_height)
        w = screen_width
        h = screen_height - 120
        self.root.geometry(f"{w}x{h}")
        
        # State variables
        self.is_calc_running = False
        self.all_points = []
        self.all_values = []
        self.selected_index = 0
        self.selected_param = None
        self.N_times = 7
        self.xx0 = None
        self.xx1 = None
        self.xx = None
        self.LL = None
        self.WA1 = 1.0
        self.WB1 = 1.0
        self.WC1 = 1.0
        self.WD1 = 1.0
        self.WA2 = 1.0
        self.WB2 = 1.0
        self.WC2 = 1.0
        self.WD2 = 1.0
        self.ww = None
        self.yy_exp = None
        self.y_max = None
        self.y_min = None
        self.BOUNDS = BOUNDS
        self.append_to_list = True
        self.filename = None
        self.filepath = None
        self.status = 'Idle'
        
        # Create UI
        self.create_widgets()
        
        # Load default file
        self.load_file(DEFAULT_FILE)

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top row with scatter plots
        top_row = ttk.Frame(main_frame)
        top_row.pack(fill=tk.BOTH, expand=True)
        
        # Scatter plots 1-4
        self.scatter_frames = []
        self.scatter_canvases = []
        self.scatter_toolbars = []

        # Main plot 1
        plot_frame_1 = ttk.LabelFrame(top_row, text="Model Fit")
        plot_frame_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_1, self.plot_ax_1 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_1 = FigureCanvasTkAgg(self.plot_fig_1, master=plot_frame_1)
        self.plot_canvas_1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_1 = NavigationToolbar2Tk(self.plot_canvas_1, plot_frame_1)
        self.plot_toolbar_1.update()

        # Main plot 2
        plot_frame_2 = ttk.LabelFrame(top_row, text="Model Fit")
        plot_frame_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_2, self.plot_ax_2 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_2 = FigureCanvasTkAgg(self.plot_fig_2, master=plot_frame_2)
        self.plot_canvas_2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_2 = NavigationToolbar2Tk(self.plot_canvas_2, plot_frame_2)
        self.plot_toolbar_2.update()

        # Main plot 3
        plot_frame_3 = ttk.LabelFrame(top_row, text="Model Fit")
        plot_frame_3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_3, self.plot_ax_3 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_3 = FigureCanvasTkAgg(self.plot_fig_3, master=plot_frame_3)
        self.plot_canvas_3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_3 = NavigationToolbar2Tk(self.plot_canvas_3, plot_frame_3)
        self.plot_toolbar_3.update()

        # Main plot 4
        plot_frame_4 = ttk.LabelFrame(top_row, text="Model Fit")
        plot_frame_4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_4, self.plot_ax_4 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_4 = FigureCanvasTkAgg(self.plot_fig_4, master=plot_frame_4)
        self.plot_canvas_4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_4 = NavigationToolbar2Tk(self.plot_canvas_4, plot_frame_4)
        self.plot_toolbar_4.update()

        # Control panel
        control_frame1 = ttk.LabelFrame(top_row, text="Controls1")
        control_frame1.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(control_frame1, text="maxiter:").pack()
        self.maxiter_entry = ttk.Entry(control_frame1)
        self.maxiter_entry.insert(0, str(DIRECT_ITER))
        self.maxiter_entry.pack(fill=tk.X)
        
        # Buttons and controls
        self.gradient_button = ttk.Button(
            control_frame1, text="Run Gradient", command=self.run_gradient_threaded
        )
        self.gradient_button.pack(fill=tk.X, pady=2)

        self.method_var = tk.StringVar()
        self.method_var.set('Nelder-Mead')
        options = ['Nelder-Mead', 'COBYLA', 'L-BFGS-B']
        tk.OptionMenu(control_frame1, self.method_var, *options).pack()

        self.skip_hessian_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_frame1, text="Skip hessian", variable=self.skip_hessian_var
        ).pack()

        self.direct_button = ttk.Button(
            control_frame1, text="Run DIRECT", command=self.run_direct_threaded
        )
        self.direct_button.pack(fill=tk.X, pady=2)
        
        # DIRECT options
        ttk.Label(control_frame1, text="epsilon:").pack()
        self.epsilon_entry = ttk.Entry(control_frame1)
        self.epsilon_entry.insert(0, str(DIRECT_EPSILON))
        self.epsilon_entry.pack(fill=tk.X)
        
        self.locally_biased_var = tk.BooleanVar(value=DIRECT_LOCALLY_BIASED)
        ttk.Checkbutton(
            control_frame1, text="Locally Biased", variable=self.locally_biased_var
        ).pack()
        
        # File upload
        self.upload_button = ttk.Button(
            control_frame1, text="Load File", command=self.upload_file
        )
        self.upload_button.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame1, text="L1:").pack()
        self.l1_entry = ttk.Entry(control_frame1)
        self.l1_entry.insert(0, str(L1))
        self.l1_entry.pack(fill=tk.X)

        ttk.Label(control_frame1, text="L2:").pack()
        self.l2_entry = ttk.Entry(control_frame1)
        self.l2_entry.insert(0, str(L2))
        self.l2_entry.pack(fill=tk.X)

        ttk.Label(control_frame1, text="L3:").pack()
        self.l3_entry = ttk.Entry(control_frame1)
        self.l3_entry.insert(0, str(L3))
        self.l3_entry.pack(fill=tk.X)

        ttk.Label(control_frame1, text="min log bound:").pack()
        self.min_log_bound_entry = ttk.Entry(control_frame1)
        self.min_log_bound_entry.insert(0, str(-5.0))
        self.min_log_bound_entry.pack(fill=tk.X)

        ttk.Label(control_frame1, text="max log bound:").pack()
        self.max_log_bound_entry = ttk.Entry(control_frame1)
        self.max_log_bound_entry.insert(0, str(1.0))
        self.max_log_bound_entry.pack(fill=tk.X)

        # File upload
        self.apply_L_and_W_button = ttk.Button(
            control_frame1, text="Apply L and W and bounds", command=self.apply_L_and_W
        )
        self.apply_L_and_W_button.pack(fill=tk.X, pady=5)

        # Control panel
        control_frame2 = ttk.LabelFrame(top_row, text="Controls1")
        control_frame2.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(control_frame2, text="WA1:").pack()
        self.wa1_entry = ttk.Entry(control_frame2)
        self.wa1_entry.insert(0, str(1.0))
        self.wa1_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WB1:").pack()
        self.wb1_entry = ttk.Entry(control_frame2)
        self.wb1_entry.insert(0, str(1.0))
        self.wb1_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WC1:").pack()
        self.wc1_entry = ttk.Entry(control_frame2)
        self.wc1_entry.insert(0, str(1.0))
        self.wc1_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WD1:").pack()
        self.wd1_entry = ttk.Entry(control_frame2)
        self.wd1_entry.insert(0, str(1.0))
        self.wd1_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WA2:").pack()
        self.wa2_entry = ttk.Entry(control_frame2)
        self.wa2_entry.insert(0, str(1.0))
        self.wa2_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WB2:").pack()
        self.wb2_entry = ttk.Entry(control_frame2)
        self.wb2_entry.insert(0, str(1.0))
        self.wb2_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WC2:").pack()
        self.wc2_entry = ttk.Entry(control_frame2)
        self.wc2_entry.insert(0, str(1.0))
        self.wc2_entry.pack(fill=tk.X)

        ttk.Label(control_frame2, text="WD2:").pack()
        self.wd2_entry = ttk.Entry(control_frame2)
        self.wd2_entry.insert(0, str(1.0))
        self.wd2_entry.pack(fill=tk.X)

        # Bottom row with more plots and info
        bottom_row = ttk.Frame(main_frame)
        bottom_row.pack(fill=tk.BOTH, expand=True)

        # Main plot 5
        plot_frame_5 = ttk.LabelFrame(bottom_row, text="Model Fit")
        plot_frame_5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_5, self.plot_ax_5 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_5 = FigureCanvasTkAgg(self.plot_fig_5, master=plot_frame_5)
        self.plot_canvas_5.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_5 = NavigationToolbar2Tk(self.plot_canvas_5, plot_frame_5)
        self.plot_toolbar_5.update()

        # Main plot 6
        plot_frame_6 = ttk.LabelFrame(bottom_row, text="Model Fit")
        plot_frame_6.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_6, self.plot_ax_6 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_6 = FigureCanvasTkAgg(self.plot_fig_6, master=plot_frame_6)
        self.plot_canvas_6.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_6 = NavigationToolbar2Tk(self.plot_canvas_6, plot_frame_6)
        self.plot_toolbar_6.update()

        # Main plot 7
        plot_frame_7 = ttk.LabelFrame(bottom_row, text="Model Fit")
        plot_frame_7.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_7, self.plot_ax_7 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_7 = FigureCanvasTkAgg(self.plot_fig_7, master=plot_frame_7)
        self.plot_canvas_7.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_7 = NavigationToolbar2Tk(self.plot_canvas_7, plot_frame_7)
        self.plot_toolbar_7.update()

        # Main plot 8
        plot_frame_8 = ttk.LabelFrame(bottom_row, text="Model Fit")
        plot_frame_8.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.plot_fig_8, self.plot_ax_8 = plt.subplots(figsize=(1.2, 1.2))
        self.plot_canvas_8 = FigureCanvasTkAgg(self.plot_fig_8, master=plot_frame_8)
        self.plot_canvas_8.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_toolbar_8 = NavigationToolbar2Tk(self.plot_canvas_8, plot_frame_8)
        self.plot_toolbar_8.update()

        # Parameter info
        info_frame = ttk.LabelFrame(bottom_row, text="Parameters")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, wrap=tk.WORD, width=50, height=15)
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind click events to scatter plots
        for i, canvas in enumerate(self.scatter_canvases):
            canvas.mpl_connect('button_press_event', lambda event, idx=i: self.on_scatter_click(event, idx))
        
        # Start update timer
        self.update_ui()

    def format_float(self, x):
        return "{:.2E}".format(x)

    def f(self, tt, LL, param):
        yy = []

        for is_ing in IS_ING:
            shift0 = 14 if is_ing else 2

            for letter in LETTERS:
                A = 10.0 **param[0]
                B = 10.0 **param[1]

                if letter == 'A':
                    shift = 0 + shift0
                elif letter == 'B':
                    shift = 3 + shift0
                elif letter == 'C':
                    shift = 6 + shift0
                else:
                    shift = 9 + shift0

                kon = 10.0 **param[0 + shift]
                koff = 10.0 **param[1 + shift]
                krel = 10.0 **param[2 + shift]

                kon_minus = 0
                koff_minus = 0
                krel_minus = 0

                for L in LL:
                    a11 = -(kon_minus + koff + koff_minus * L)
                    a12 = kon * L - koff_minus * L
                    a21 = kon_minus - krel
                    a22 = -(krel + krel_minus + kon * L)

                    b1 = koff_minus * L
                    b2 = krel

                    def g(y, t):
                        y1 = y[0]
                        y2 = y[1]
                        dy1 = a11 * y1 + a12 * y2 + b1
                        dy2 = a21 * y1 + a22 * y2 + b2
                        return [dy1, dy2]

                    g0 = [0, krel / (krel + krel_minus)]

                    sol = odeint(g, g0, [0] + tt.tolist())
                    sol = sol.tolist()
                    sol.pop(0)
                    for s in sol:
                        X = s[0]
                        R_a = s[1]

                        y = B + A * X
                        yy.append(y)

        return yy

    def ff(self, param):
        yy0 = np.array(self.f(self.xx, self.LL, param))
        
        dy = abs(self.yy_exp - yy0) ** 2
        # dy = abs(self.yy_exp - yy0) ** 2 / self.yy_exp
        # dy = abs(self.yy_exp - yy0) ** 4 * 1e-4
        # dy = abs(self.yy_exp - yy0)
        # dy = dy / abs(self.yy_exp) ** 2
        dy = dy * self.ww ** 2
        ssq = np.sum(dy)
        (n1,) = self.xx.shape
        (n2,) = self.LL.shape
        n3 = len(LETTERS)
        n = n1 * n2 * n3
        sq = ssq / n

        if self.append_to_list:
            self.all_points.append(param.copy())
            self.all_values.append(sq)
            print(sq, len(self.all_values))
            if len(self.all_values) % 2500 == 0:
                self.selected_index = np.argmin(self.all_values)
                self.selected_param = self.all_points[self.selected_index]
                self.update_plots()

        return sq

    def load_file(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        try:
            with open(filepath, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                headers = next(reader)
                data = np.array(list(reader))
                data = data.astype(float)

            (N_times, N_columns) = data.shape
            self.N_times = N_times
            self.xx0 = data[:, 0]
            self.xx1 = np.arange(self.xx0[0], self.xx0[-1], 0.1)

            _L1 = float(self.l1_entry.get())
            _L2 = float(self.l2_entry.get())
            _L3 = float(self.l3_entry.get())
            self.LL = np.array([_L1, _L2, _L3])
            print(self.LL)
            self.xx = np.array(self.xx0)

            min_bound = float(self.min_log_bound_entry.get())
            max_bound = float(self.max_log_bound_entry.get())
            for i in np.arange(2, 26):
                self.BOUNDS[i] = (min_bound, max_bound)
            print(self.BOUNDS)

            self.WA1 = float(self.wa1_entry.get())
            self.WB1 = float(self.wb1_entry.get())
            self.WC1 = float(self.wc1_entry.get())
            self.WD1 = float(self.wd1_entry.get())
            self.WA2 = float(self.wa2_entry.get())
            self.WB2 = float(self.wb2_entry.get())
            self.WC2 = float(self.wc2_entry.get())
            self.WD2 = float(self.wd2_entry.get())

            self.ww = np.array([self.WA1] * N_times * 3 + [self.WB1] * N_times * 3 +
                               [self.WC1] * N_times * 3 + [self.WD1] * N_times * 3 +
                               [self.WA2] * N_times * 3 + [self.WB2] * N_times * 3 +
                               [self.WC2] * N_times * 3 + [self.WD2] * N_times * 3)
            print(self.ww)

            self.yy_exp = np.concatenate((data[:, 1], data[:, 2], data[:, 3],
                                          data[:, 7], data[:, 8], data[:, 9],
                                          data[:, 13], data[:, 14], data[:, 15],
                                          data[:, 19], data[:, 20], data[:, 21],
                                          data[:, 4], data[:, 5], data[:, 6],
                                          data[:, 10], data[:, 11], data[:, 12],
                                          data[:, 16], data[:, 17], data[:, 18],
                                          data[:, 22], data[:, 23], data[:, 24],))

            self.y_max = np.max(self.yy_exp)
            self.y_min = np.min(self.yy_exp)

            self.all_points = []
            self.all_values = []
            self.selected_index = 0

            self.selected_param = [log10(3 * self.y_max), log10(300),
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1]
            self.ff(self.selected_param)
            
            self.update_plots()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            return False

    def upload_file(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            if self.load_file(filepath):
                messagebox.showinfo("Success", f"File {os.path.basename(filepath)} loaded successfully")

    def apply_L_and_W(self):
        self.load_file(self.filepath)

    def run_direct_threaded(self):
        if self.is_calc_running:
            return
        
        maxiter = int(self.maxiter_entry.get())
        epsilon = float(self.epsilon_entry.get())
        locally_biased = self.locally_biased_var.get()
        
        thread = threading.Thread(
            target=self.run_direct,
            args=(maxiter, epsilon, locally_biased),
            daemon=True
        )
        thread.start()

    def run_direct(self, maxiter, epsilon, locally_biased):
        self.all_points = []
        self.all_values = []
        self.selected_index = 0
        self.selected_param = [3.5, log10(300),
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1,
                                   -1, -1, -1]

        self.ff(self.selected_param)

        self.is_calc_running = True
        self.status = 'Running direct'
        self.update_plots()
        self.update_button_states()

        result = direct(
            self.ff,
            maxfun=maxiter,
            maxiter=maxiter,
            eps=epsilon,
            locally_biased=locally_biased,
            bounds=self.BOUNDS
        )

        self.selected_index = np.argmin(self.all_values)
        self.selected_param = self.all_points[self.selected_index]
        self.is_calc_running = False
        self.status = 'Idle'
        self.update_plots()
        self.update_button_states()

    def run_gradient_threaded(self):
        if self.is_calc_running:
            return

        maxiter = int(self.maxiter_entry.get())
        method = self.method_var.get()

        thread = threading.Thread(
            target=self.run_gradient,
            args=(maxiter, method),
            daemon=True
        )
        thread.start()

    def print_param_with_errors(self, param, dP):
        with open(self.filename + "_fit10.txt", "w") as fn:
            for i in range(len(PARAM_NAMES)):
                param_name = PARAM_NAMES[i]
                p = param[i]
                dp = dP[i]
                if param_name.startswith('alpha'):
                    print(param_name,
                          self.format_float((erf(p) + 1) / 2),
                          self.format_float((erf(p - dp) + 1) / 2),
                          self.format_float((erf(p + dp) + 1) / 2))
                    fn.write('{0} {1} {2} {3}\n'.format(
                             param_name,
                             self.format_float((erf(p) + 1) / 2),
                             self.format_float((erf(p - dp) + 1) / 2),
                             self.format_float((erf(p + dp) + 1) / 2)))
                else:
                    is_zero = param_name.startswith('kon_minus') and self.zero_kon_minus or \
                                param_name.startswith('koff_minus') and self.zero_koff_minus or \
                                param_name.startswith('krel_minus') and self.zero_krel_minus
                    print(param_name,
                          self.format_float(10.0 ** p if not is_zero else 0),
                          self.format_float(10.0 ** (p - dp) if not is_zero else 0),
                          self.format_float(10.0 ** (p + dp) if not is_zero else 0))
                    fn.write('{0} {1} {2} {3}\n'.format(
                        param_name,
                        self.format_float(10.0 ** p if not is_zero else 0),
                        self.format_float(10.0 ** (p - dp) if not is_zero else 0),
                        self.format_float(10.0 ** (p + dp) if not is_zero else 0)))

    def save_plot_to_png(self, param):
        yy0 = np.array(
            self.f(self.xx1, self.LL, param))

        (N_times_0, ) = self.xx1.shape

        plt.figure()
        fig, axs = plt.subplots(2, 4, figsize=(18, 12))
        fig.suptitle(self.filename)
        axs[0, 0].plot(self.xx1, yy0[N_times_0*0:N_times_0*1], c='blue')
        axs[0, 0].plot(self.xx1, yy0[N_times_0*1:N_times_0*2], c='red')
        axs[0, 0].plot(self.xx1, yy0[N_times_0*2:N_times_0*3], c='green')
        axs[0, 0].plot(self.xx0, self.yy_exp[self.N_times*0:self.N_times*1], '--', c='blue')
        axs[0, 0].plot(self.xx0, self.yy_exp[self.N_times*1:self.N_times*2], '--', c='red')
        axs[0, 0].plot(self.xx0, self.yy_exp[self.N_times*2:self.N_times*3], '--', c='green')
        axs[0, 1].plot(self.xx1, yy0[N_times_0 * 3:N_times_0 * 4], c='blue')
        axs[0, 1].plot(self.xx1, yy0[N_times_0 * 4:N_times_0 * 5], c='red')
        axs[0, 1].plot(self.xx1, yy0[N_times_0 * 5:N_times_0 * 6], c='green')
        axs[0, 1].plot(self.xx0, self.yy_exp[self.N_times * 3:self.N_times * 4], '--', c='blue')
        axs[0, 1].plot(self.xx0, self.yy_exp[self.N_times * 4:self.N_times * 5], '--', c='red')
        axs[0, 1].plot(self.xx0, self.yy_exp[self.N_times * 5:self.N_times * 6], '--', c='green')
        axs[0, 2].plot(self.xx1, yy0[N_times_0 * 6:N_times_0 * 7], c='blue')
        axs[0, 2].plot(self.xx1, yy0[N_times_0 * 7:N_times_0 * 8], c='red')
        axs[0, 2].plot(self.xx1, yy0[N_times_0 * 8:N_times_0 * 9], c='green')
        axs[0, 2].plot(self.xx0, self.yy_exp[self.N_times * 6:self.N_times * 7], '--', c='blue')
        axs[0, 2].plot(self.xx0, self.yy_exp[self.N_times * 7:self.N_times * 8], '--', c='red')
        axs[0, 2].plot(self.xx0, self.yy_exp[self.N_times * 8:self.N_times * 9], '--', c='green')
        axs[0, 3].plot(self.xx1, yy0[N_times_0 * 9:N_times_0 * 10], c='blue')
        axs[0, 3].plot(self.xx1, yy0[N_times_0 * 10:N_times_0 * 11], c='red')
        axs[0, 3].plot(self.xx1, yy0[N_times_0 * 11:N_times_0 * 12], c='green')
        axs[0, 3].plot(self.xx0, self.yy_exp[self.N_times * 9:self.N_times * 10], '--', c='blue')
        axs[0, 3].plot(self.xx0, self.yy_exp[self.N_times * 10:self.N_times * 11], '--', c='red')
        axs[0, 3].plot(self.xx0, self.yy_exp[self.N_times * 11:self.N_times * 12], '--', c='green')
        axs[1, 0].plot(self.xx1, yy0[N_times_0 * 12:N_times_0 * 13], c='blue')
        axs[1, 0].plot(self.xx1, yy0[N_times_0 * 13:N_times_0 * 14], c='red')
        axs[1, 0].plot(self.xx1, yy0[N_times_0 * 14:N_times_0 * 15], c='green')
        axs[1, 0].plot(self.xx0, self.yy_exp[self.N_times * 12:self.N_times * 13], '--', c='blue')
        axs[1, 0].plot(self.xx0, self.yy_exp[self.N_times * 13:self.N_times * 14], '--', c='red')
        axs[1, 0].plot(self.xx0, self.yy_exp[self.N_times * 14:self.N_times * 15], '--', c='green')
        axs[1, 1].plot(self.xx1, yy0[N_times_0 * 15:N_times_0 * 16], c='blue')
        axs[1, 1].plot(self.xx1, yy0[N_times_0 * 16:N_times_0 * 17], c='red')
        axs[1, 1].plot(self.xx1, yy0[N_times_0 * 17:N_times_0 * 18], c='green')
        axs[1, 1].plot(self.xx0, self.yy_exp[self.N_times * 15:self.N_times * 16], '--', c='blue')
        axs[1, 1].plot(self.xx0, self.yy_exp[self.N_times * 16:self.N_times * 17], '--', c='red')
        axs[1, 1].plot(self.xx0, self.yy_exp[self.N_times * 17:self.N_times * 18], '--', c='green')
        axs[1, 2].plot(self.xx1, yy0[N_times_0 * 18:N_times_0 * 19], c='blue')
        axs[1, 2].plot(self.xx1, yy0[N_times_0 * 19:N_times_0 * 20], c='red')
        axs[1, 2].plot(self.xx1, yy0[N_times_0 * 20:N_times_0 * 21], c='green')
        axs[1, 2].plot(self.xx0, self.yy_exp[self.N_times * 18:self.N_times * 19], '--', c='blue')
        axs[1, 2].plot(self.xx0, self.yy_exp[self.N_times * 19:self.N_times * 20], '--', c='red')
        axs[1, 2].plot(self.xx0, self.yy_exp[self.N_times * 20:self.N_times * 21], '--', c='green')
        axs[1, 3].plot(self.xx1, yy0[N_times_0 * 21:N_times_0 * 22], c='blue')
        axs[1, 3].plot(self.xx1, yy0[N_times_0 * 22:N_times_0 * 23], c='red')
        axs[1, 3].plot(self.xx1, yy0[N_times_0 * 23:N_times_0 * 24], c='green')
        axs[1, 3].plot(self.xx0, self.yy_exp[self.N_times * 21:self.N_times * 22], '--', c='blue')
        axs[1, 3].plot(self.xx0, self.yy_exp[self.N_times * 22:self.N_times * 23], '--', c='red')
        axs[1, 3].plot(self.xx0, self.yy_exp[self.N_times * 23:self.N_times * 24], '--', c='green')
        plt.savefig(self.filename + '_fit10.png')

    def save_derived_plot_to_png(self, param, kon_minus_coeff, krel_coeff, L_coeff, fn):
        yy0 = np.array(
            self.f(self.xx1, self.LL * L_coeff, param))

        (N_times_0,) = self.xx1.shape

        max1 = yy0[N_times_0 * 0:N_times_0 * 1][argrelextrema(yy0[N_times_0 * 0:N_times_0 * 1], np.greater)]
        max2 = yy0[N_times_0 * 1:N_times_0 * 2][argrelextrema(yy0[N_times_0 * 1:N_times_0 * 2], np.greater)]
        max3 = yy0[N_times_0 * 2:N_times_0 * 3][argrelextrema(yy0[N_times_0 * 2:N_times_0 * 3], np.greater)]
        print('kon_minus: {0} krel: {1} L:{2} max1: {3} max2: {4} max3: {5}'.format(kon_minus_coeff, krel_coeff, L_coeff, max1, max2, max3))
        fn.write(
            'kon_minus: {0} krel: {1} L:{2} max1: {3} max2: {4} max3: {5}\n'.format(kon_minus_coeff, krel_coeff, L_coeff, max1,
                                                                               max2, max3))

        plt.figure()
        plt.title(self.filename + '\nkon_minus: {0} krel: {1} L:{2}'.format(kon_minus_coeff, krel_coeff, L_coeff))
        plt.plot(self.xx1, yy0[N_times_0 * 0:N_times_0 * 1], c='blue')
        plt.plot(self.xx1, yy0[N_times_0 * 1:N_times_0 * 2], c='red')
        plt.plot(self.xx1, yy0[N_times_0 * 2:N_times_0 * 3], c='green')
        plt.savefig('derived/' + self.filename + '_kon_minus_{0}_krel_{1}_L_{2}.png'.format(kon_minus_coeff, krel_coeff, L_coeff))

        if len(max1) > 0:
            return max1[0]
        else:
            return np.nan

    def run_gradient(self, maxiter, method):
        print(method, maxiter)

        self.status = 'Running gradient'
        self.is_calc_running = True
        self.update_plots()
        self.update_button_states()

        _W = (self.WA1 ** 2 + self.WB1 ** 2 + self.WC1 ** 2 + self.WD1 ** 2) ** 0.5

        result = minimize(
            self.ff,
            method=method,
            x0=self.selected_param,
            bounds=self.BOUNDS,
            options={'maxiter': maxiter, 'fatol': (self.y_max * _W / 1000.0), 'tol': 1e-4}
        )
        print(result)
        param = result.x

        self.append_to_list = False

        self.selected_index = len(self.all_points) - 1
        self.selected_param = self.all_points[self.selected_index]

        skip_hessian = self.skip_hessian_var.get()

        if not skip_hessian:
            try:
                self.status = 'Calculating hessian'
                self.update_plots()
                self.update_button_states()

                # raise Exception('skip')
                hessian = Hessian(self.ff)(param)
                print('hessian', hessian)

                # Compute covariance matrix (inverse of Hessian)
                cov_matrix = np.linalg.inv(hessian)
                dP = np.sqrt(np.diag(cov_matrix))
                print('dP', dP)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to calculate errors: {str(e)}")
                dP = [np.nan] * len(param)
        else:
            dP = [np.nan] * len(param)

        self.status = 'Saving results'
        self.print_param_with_errors(param, dP)
        self.save_plot_to_png(param)

        # L_coeffs_log = np.arange(log10(0.0002), log10(2.0) + 0.2, 0.2)
        # maxx = []
        # with open('derived_' + self.filename + ".txt", "w") as fn:
        #     for L_coeff_log in L_coeffs_log:
        #         maxx.append(self.save_derived_plot_to_png(param, 1.0, 1.0, 10.0 **L_coeff_log, fn))
        #     # self.save_derived_plot_to_png(param, 0.1, 0.1, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 0.1, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 0.1, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 1.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 1.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 1.0, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 10.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 10.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 0.1, 10.0, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 0.1, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 0.1, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 0.1, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 1.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 1.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 1.0, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 10.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 10.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 1.0, 10.0, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 0.1, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 0.1, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 0.1, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 1.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 1.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 1.0, 2.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 10.0, 0.5, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 10.0, 1.0, fn)
        #     # self.save_derived_plot_to_png(param, 10.0, 10.0, 2.0, fn)
        #
        # Ll = 1.0 + L_coeffs_log
        # maxx = np.array(maxx)
        # plt.figure()
        # plt.plot(Ll, maxx)
        # plt.savefig('derived_maxx_' + self.filename + '.png')

        self.append_to_list = True
        self.is_calc_running = False
        self.status = 'Idle'
        self.update_plots()
        self.update_button_states()

    def update_button_states(self):
        state = tk.DISABLED if self.is_calc_running else tk.NORMAL
        self.gradient_button.config(state=state)
        self.direct_button.config(state=state)
        self.upload_button.config(state=state)
        self.apply_L_and_W_button.config(state=state)

    def on_scatter_click(self, event, plot_idx):
        if not event.inaxes:
            return
            
        # Find closest point
        if len(self.all_points) == 0:
            return
            
        # Determine which parameters this plot shows
        param_pairs = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (8, 9), (10, 11), (12, 13)
        ]
        if plot_idx >= len(param_pairs):
            return
            
        x_idx, y_idx = param_pairs[plot_idx]
        
        # Get all points in this parameter space
        x_vals = [p[x_idx] for p in self.all_points]
        y_vals = [p[y_idx] for p in self.all_points]
        
        # Find closest point to click
        distances = [
            (event.xdata - x)**2 + (event.ydata - y)**2 
            for x, y in zip(x_vals, y_vals)
        ]
        closest_idx = np.argmin(distances)
        
        self.selected_index = closest_idx
        self.selected_param = self.all_points[self.selected_index]
        self.update_plots()

    def update_plots(self):
        if len(self.all_points) <= self.selected_index:
            return

        # self.selected_param = self.all_points[self.selected_index]
        param = self.selected_param

        # Update parameter info
        A = 10.0 **param[0]
        B = 10.0 **param[1]
        shift = 2
        kon_A1 = 10.0 **param[0 + shift]
        koff_A1 = 10.0 **param[1 + shift]
        krel_A1 = 10.0 ** param[2 + shift]
        shift = 5
        kon_B1 = 10.0 **param[0 + shift]
        koff_B1 = 10.0 **param[1 + shift]
        krel_B1 = 10.0 ** param[2 + shift]
        shift = 8
        kon_C1 = 10.0 **param[0 + shift]
        koff_C1 = 10.0 **param[1 + shift]
        krel_C1 = 10.0 ** param[2 + shift]
        shift = 11
        kon_D1 = 10.0 **param[0 + shift]
        koff_D1 = 10.0 **param[1 + shift]
        krel_D1 = 10.0 ** param[2 + shift]
        shift = 14
        kon_A2 = 10.0 ** param[0 + shift]
        koff_A2 = 10.0 ** param[1 + shift]
        krel_A2 = 10.0 ** param[2 + shift]
        shift = 17
        kon_B2 = 10.0 ** param[0 + shift]
        koff_B2 = 10.0 ** param[1 + shift]
        krel_B2 = 10.0 ** param[2 + shift]
        shift = 20
        kon_C2 = 10.0 ** param[0 + shift]
        koff_C2 = 10.0 ** param[1 + shift]
        krel_C2 = 10.0 ** param[2 + shift]
        shift = 23
        kon_D2 = 10.0 ** param[0 + shift]
        koff_D2 = 10.0 ** param[1 + shift]
        krel_D2 = 10.0 ** param[2 + shift]
        value = self.ff(param)
        
        info_text = f"""
status: {self.status}
iter = {len(self.all_points)}
value = {self.format_float(value)}
A = {self.format_float(A)}
B = {self.format_float(B)}
kon_A1 = {self.format_float(kon_A1)}
koff_A1 = {self.format_float(koff_A1)}
krel_A1 = {self.format_float(krel_A1)}
kon_B1 = {self.format_float(kon_B1)}
koff_B1 = {self.format_float(koff_B1)}
krel_B1 = {self.format_float(krel_B1)}
kon_C1 = {self.format_float(kon_C1)}
koff_C1 = {self.format_float(koff_C1)}
krel_C1 = {self.format_float(krel_C1)}
kon_D1 = {self.format_float(kon_D1)}
koff_D1 = {self.format_float(koff_D1)}
krel_D1 = {self.format_float(krel_D1)}
kon_A2 = {self.format_float(kon_A2)}
koff_A2 = {self.format_float(koff_A2)}
krel_A2 = {self.format_float(krel_A2)}
kon_B2 = {self.format_float(kon_B2)}
koff_B2 = {self.format_float(koff_B2)}
krel_B2 = {self.format_float(krel_B2)}
kon_C2 = {self.format_float(kon_C2)}
koff_C2 = {self.format_float(koff_C2)}
krel_C2 = {self.format_float(krel_C2)}
kon_D2 = {self.format_float(kon_D2)}
koff_D2 = {self.format_float(koff_D2)}
krel_D2 = {self.format_float(krel_D2)}
"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text.strip())
        
        _all_points = np.array(self.all_points)
        _all_values = np.array(self.all_values)
        (N,) = _all_values.shape
        colors = np.array(['blue'] * N)
        colors[self.selected_index] = 'red'
        sizes = np.array([20] * N)
        sizes[self.selected_index] = 100

        # Update main plot
        yy0 = np.array(self.f(self.xx1, self.LL, param))

        (N_times_0,) = self.xx1.shape

        self.plot_fig_1.clf()
        ax = self.plot_fig_1.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 0:N_times_0 * 1], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 1:N_times_0 * 2], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 2:N_times_0 * 3], c='green')
        
        ax.plot(self.xx0, self.yy_exp[self.N_times*0:self.N_times*1], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times*1:self.N_times*2], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times*2:self.N_times*3], '--', c='green')

        ax.set_title("Model Fit A1")
        self.plot_canvas_1.draw()

        self.plot_fig_2.clf()
        ax = self.plot_fig_2.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 3:N_times_0 * 4], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 4:N_times_0 * 5], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 5:N_times_0 * 6], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 3:self.N_times * 4], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 4:self.N_times * 5], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 5:self.N_times * 6], '--', c='green')

        ax.set_title("Model Fit B1")
        self.plot_canvas_2.draw()

        self.plot_fig_3.clf()
        ax = self.plot_fig_3.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 6:N_times_0 * 7], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 7:N_times_0 * 8], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 8:N_times_0 * 9], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 6:self.N_times * 7], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 7:self.N_times * 8], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 8:self.N_times * 9], '--', c='green')

        ax.set_title("Model Fit C1")
        self.plot_canvas_3.draw()

        self.plot_fig_4.clf()
        ax = self.plot_fig_4.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 9:N_times_0 * 10], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 10:N_times_0 * 11], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 11:N_times_0 * 12], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 9:self.N_times * 10], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 10:self.N_times * 11], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 11:self.N_times * 12], '--', c='green')

        ax.set_title("Model Fit D1")
        self.plot_canvas_4.draw()

        self.plot_fig_5.clf()
        ax = self.plot_fig_5.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 12:N_times_0 * 13], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 13:N_times_0 * 14], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 14:N_times_0 * 15], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 12:self.N_times * 13], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 13:self.N_times * 14], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 14:self.N_times * 15], '--', c='green')

        ax.set_title("Model Fit A2")
        self.plot_canvas_5.draw()

        self.plot_fig_6.clf()
        ax = self.plot_fig_6.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 15:N_times_0 * 16], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 16:N_times_0 * 17], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 17:N_times_0 * 18], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 15:self.N_times * 16], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 16:self.N_times * 17], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 17:self.N_times * 18], '--', c='green')

        ax.set_title("Model Fit B2")
        self.plot_canvas_6.draw()

        self.plot_fig_7.clf()
        ax = self.plot_fig_7.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 18:N_times_0 * 19], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 19:N_times_0 * 20], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 20:N_times_0 * 21], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 18:self.N_times * 19], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 19:self.N_times * 20], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 20:self.N_times * 21], '--', c='green')

        ax.set_title("Model Fit C2")
        self.plot_canvas_7.draw()

        self.plot_fig_8.clf()
        ax = self.plot_fig_8.add_subplot(111)

        ax.plot(self.xx1, yy0[N_times_0 * 21:N_times_0 * 22], c='blue')
        ax.plot(self.xx1, yy0[N_times_0 * 22:N_times_0 * 23], c='red')
        ax.plot(self.xx1, yy0[N_times_0 * 23:N_times_0 * 24], c='green')

        ax.plot(self.xx0, self.yy_exp[self.N_times * 21:self.N_times * 22], '--', c='blue')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 22:self.N_times * 23], '--', c='red')
        ax.plot(self.xx0, self.yy_exp[self.N_times * 23:self.N_times * 24], '--', c='green')

        ax.set_title("Model Fit D2")
        self.plot_canvas_8.draw()

    def update_ui(self):
        self.update_plots()
        # self.root.after(2000, self.update_ui)  # Update every 2 seconds

if __name__ == "__main__":
    root = tk.Tk()
    app = KineticModelApp(root)
    root.mainloop()