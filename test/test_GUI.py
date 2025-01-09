import sys
sys.path.append('./')

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from PIL import Image, ImageTk

from src.genetic_algorithm import GeneticAlgorithm
from src.minima_hopping_optimizer import MinimaHoppingOptimizer
from src.basin_hopping_optimizer import BasinHoppingOptimizer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ase.visualize.plot import plot_atoms
from ase.visualize import view

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('Agg')

#import ttkbootstrap as ttk
#style = ttk.Style("darkly")

class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Atomic Materials Structure Optimizer")
        self.root.geometry("800x600")
        self.start_frame = tk.Frame(self.root)
        self.start_frame.pack(expand=True, fill="both")
        self.create_start()

    def set_background(self, image_path):
            # Load the image
            image = Image.open(image_path)
            self.background_image = ImageTk.PhotoImage(image)

            # Display the image
            background_label = tk.Label(self.start_frame, image=self.background_image)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)

    def create_start(self):
        self.set_background("../../../Downloads/imagebg.webp")

        menu_frame = tk.Frame(self.start_frame, bd=0)
        menu_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Add menu buttons
        button_style = {
            "bg": "#f0f0f0",
            "fg": "#333333",
            "relief": "flat",
            "bd": 0,
            "highlightthickness": 0,
            "font": ("Arial", 12, "bold"),
            "activebackground": "#e0e0e0",
            "activeforeground": "#000000",}

        tk.Button(menu_frame, text="Start Simulation",**button_style,
                  command=self.start_simulation).pack(pady=10)
        tk.Button(menu_frame, text="Settings", **button_style,
                  command=self.settings).pack(pady=10)
        tk.Button(menu_frame, text="Quit", **button_style,
                  command=self.root.quit).pack(pady=10)
        
    def destroy_start(self):
        self.start_frame.pack_forget()
    
    def start_simulation(self):
        self.destroy_start()
        self.create_menu()
        self.simulation_frame = tk.Frame(self.root)
        self.simulation_frame.pack(expand=True)

        ttk.Label(self.simulation_frame, text="Atomic Materials Structure Optimizer", font=("Arial", 16)).pack(pady=10)

        ttk.Label(self.simulation_frame, text="Select Optimization Method:").pack()
        self.optimizer_var = tk.StringVar()
        self.optimizer_dropdown = ttk.Combobox(self.simulation_frame, textvariable=self.optimizer_var, state="readonly")
        self.optimizer_dropdown['values'] = ("Genetic Algorithm", "Minima Hopping", "Basin Hopping")
        self.optimizer_dropdown.current(0)
        self.optimizer_dropdown.pack(pady=5)
        

        ttk.Label(self.simulation_frame, text="Select Interaction/Calculator:").pack()
        self.calculator_var = tk.StringVar()
        self.calculator_dropdown = ttk.Combobox(self.simulation_frame, textvariable=self.calculator_var, state="readonly")
        self.calculator_dropdown['values'] = ("LJ", "EMT", "Tersoff", "Stillinger-Weber")
        self.calculator_dropdown.current(0)
        self.calculator_dropdown.pack(pady=5)

        self.create_input_fields()

        self.run_button = ttk.Button(self.simulation_frame, text="Run Optimizer", command=self.run_optimizer)
        self.run_button.pack(pady=10)

        self.log_text = None
        self.fig, self.ax = None, None
        self.canvas = None

        # State tracking for views
        self.graph_shown = False
        self.structure_shown = False


    def settings(self):
        tk.messagebox.showinfo("Settings", "Open settings menu.")

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Graph", command=self.show_graph)
        view_menu.add_command(label="Structure", command=self.show_structure)
        view_menu.add_command(label="3D Interactive Model", command=self.show_3d_model)
        view_menu.add_separator()
        view_menu.add_command(label="Log Output", command=self.show_log)

    def create_input_fields(self):
        self.input_frame = tk.Frame(self.simulation_frame)
        self.input_frame.pack()

        ttk.Label(self.input_frame, text="Number of Atoms:").grid(row=0, column=0, padx=5, pady=5)
        self.num_atoms_var = tk.IntVar(value=13)
        self.num_atoms_entry = ttk.Entry(self.input_frame, textvariable=self.num_atoms_var, width=10)
        self.num_atoms_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Element (e.g., C):").grid(row=1, column=0, padx=5, pady=5)
        self.element_var = tk.StringVar(value="C")
        self.element_entry = ttk.Entry(self.input_frame, textvariable=self.element_var, width=10)
        self.element_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Max Number of Iterations:").grid(row=2, column=0, padx=5, pady=5)
        self.num_iter_var = tk.IntVar(value=50)
        self.num_iter_entry = ttk.Entry(self.input_frame, textvariable=self.num_iter_var, width=10)
        self.num_iter_entry.grid(row=2, column=1, padx=5, pady=5)

    def run_optimizer(self):
        try:
            optimizer_choice = self.optimizer_var.get()
            calculator_choice = self.calculator_var.get()
            num_atoms = self.num_atoms_var.get()
            element = self.element_var.get()
            iterations = self.num_iter_var.get()

            self.log(f"Running {optimizer_choice} with {calculator_choice} potential: num_atoms={num_atoms}, Element='{element}', iterations={iterations}")
            calculator_ = self.get_calculator(calculator_choice)

            if optimizer_choice == "Genetic Algorithm":
                ga = GeneticAlgorithm(num_clusters=4, preserve=True, calculator=calculator_)
                ga.run(num_atoms, element, iterations)
                self.myatoms = ga.cluster_list[-1]
                self.log("Genetic Algorithm completed successfully.")
                self.plot_trajectory(ga.potentials)
                self.show_log()

            elif optimizer_choice == "Minima Hopping":
                mh = MinimaHoppingOptimizer(calculator=calculator_)
                mh.run(num_atoms, element, iterations)
                self.myatoms = mh.cluster_list[-1]
                self.log("Minima Hopping completed successfully.")
                self.plot_trajectory(mh.potentials)
                self.show_log()

            elif optimizer_choice == "Basin Hopping":
                bh = BasinHoppingOptimizer(calculator=calculator_)
                bh.run(num_atoms, element, iterations)
                self.myatoms = bh.cluster_list[-1]
                self.log("Basin Hopping completed successfully.")
                self.plot_trajectory(bh.potentials)
                self.show_log()

            else:
                messagebox.showerror("Error", "Invalid optimizer selection.")

        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_calculator(self, calculator_choice):
        if calculator_choice == "LJ":
            from ase.calculators.lj import LennardJones
            return LennardJones
        elif calculator_choice == "EMT":
            from ase.calculators.emt import EMT
            return EMT
        elif calculator_choice == "Tersoff":
            from ase.calculators.lammps import LAMMPS
            return LAMMPS(parameters={"pair_style": "tersoff", "pair_coeff": ["* * SiC.tersoff Si C"]})
        elif calculator_choice == "Stillinger-Weber":
            from ase.calculators.lammps import LAMMPS
            return LAMMPS(parameters={"pair_style": "sw", "pair_coeff": ["* * SiC.sw Si C"]})
        else:
            raise ValueError("Unsupported calculator.")

    def log(self, message):
        if not self.log_text:
            self.log_text = tk.Text(self.simulation_frame, height=10, width=60)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def plot_trajectory(self, potentials):
        self.hide_all_views()
        if not self.fig or not self.canvas:
            self.fig, self.ax = plt.subplots(figsize=(5, 4))
            self.canvas = FigureCanvasTkAgg(self.fig, self.simulation_frame)

        self.ax.clear()
        self.ax.plot(range(len(potentials)), potentials)
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Energy')
        self.ax.set_title('Optimization Trajectory')
        self.canvas.draw()

    def show_graph(self):
        self.hide_all_views()
        if self.canvas:
            self.canvas.get_tk_widget().pack(pady=10)

    def show_structure(self):
        self.hide_all_views()
    
        if not self.fig or not self.ax:
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 4))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.simulation_frame)

        self.ax.clear()  
        plot_atoms(self.myatoms, ax=self.ax, radii=0.5)  
        self.ax.set_title("3D Structure View")

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(pady=10)
        self.canvas.draw()

    def show_3d_model(self):
        self.hide_all_views()
        view(self.myatoms)

    def show_log(self):
        self.hide_all_views()
        if self.log_text:
            self.log_text.pack(pady=5)

    def hide_all_views(self):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        if self.log_text:
            self.log_text.pack_forget()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()
