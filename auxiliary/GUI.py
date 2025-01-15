import sys
sys.path.append('./')

import os

import numpy as np
import tkinter as tk
import ase
from tkinter import ttk
from tkinter import messagebox

from PIL import Image, ImageTk

from collections import OrderedDict

from ase.io import Trajectory

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

from auxiliary.gpw import gpw

class OptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Atomic Materials Structure Optimizer")
        self.root.geometry("800x600")
        self.start_frame = tk.Frame(self.root)
        self.start_frame.pack(expand=True, fill="both")
        self.create_start()

    def set_background(self, image_path):
            image = Image.open(image_path)
            self.background_image = ImageTk.PhotoImage(image)

            background_label = tk.Label(self.start_frame, image=self.background_image)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)

    def create_start(self):
        self.set_background("static/imagebg.webp")

        menu_frame = tk.Frame(self.start_frame, bd=0)
        menu_frame.place(relx=0.5, rely=0.5, anchor="center")

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
        self.optimizer_dropdown['values'] = (" ","Genetic Algorithm", "Basin Hopping")
        self.optimizer_dropdown.current(0)
        self.optimizer_dropdown.pack(pady=5)
        

        ttk.Label(self.simulation_frame, text="Select Interaction/Calculator:").pack()
        self.calculator_var = tk.StringVar()
        self.calculator_dropdown = ttk.Combobox(self.simulation_frame, textvariable=self.calculator_var, state="readonly")
        self.calculator_dropdown['values'] = (" ","Lennard Jones", "EMT","EAM" , "GPAW")
        self.calculator_dropdown.current(0)
        self.calculator_dropdown.pack(pady=5)

        self.create_input_fields()
        
        self.optimizer_dropdown.bind("<<ComboboxSelected>>", lambda event: self.selected_optimizer())


        self.run_button = ttk.Button(self.simulation_frame, text="Run Optimizer", command=self.run_optimizer)
        self.run_button.pack(pady=10)

        self.log_text = None
        self.fig, self.ax = None, None
        self.canvas = None

        # State tracking for views
        self.graph_shown = False

    def settings(self):
        tk.messagebox.showinfo("Settings", "Open settings menu.")

    def create_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        self.view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Graph", command=self.show_graph)
        self.view_menu.add_command(label="3D Interactive Model", command=self.show_3d_model)
        self.view_menu.add_separator()
        self.view_menu.add_command(label="Log Output", command=self.show_log)

    def create_input_fields(self):
        self.input_frame = tk.Frame(self.simulation_frame)
        self.input_frame.pack()

        ttk.Label(self.input_frame, text="Atoms (e.g., C2H4 for 2 carbons and 4 hydrogens):").grid(row=1, column=0, padx=5, pady=5)
        self.element_var = tk.StringVar(value="C2H4")
        self.element_entry = ttk.Entry(self.input_frame, textvariable=self.element_var, width=10)
        self.element_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Max Number of Iterations:").grid(row=2, column=0, padx=5, pady=5)
        self.num_iter_var = tk.IntVar(value=50)
        self.num_iter_entry = ttk.Entry(self.input_frame, textvariable=self.num_iter_var, width=10)
        self.num_iter_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(self.input_frame, text="Min Number of Iterations to convergence:").grid(row=3, column=0, padx=5, pady=5)
        self.conv_iter_var = tk.IntVar(value=10)
        self.conv_iter_entry = ttk.Entry(self.input_frame, textvariable=self.conv_iter_var, width=10)
        self.conv_iter_entry.grid(row=3, column=1, padx=5, pady=5)
    
    def selected_optimizer(self):
        if self.optimizer_var.get() == "Genetic Algorithm":
            ttk.Label(self.input_frame, text="Mutation Parameters:").grid(row=4, column=0, padx=5, pady=5)
            self.mutation_var = tk.StringVar(value="Default")
            self.mutation_dropdown = ttk.Combobox(self.input_frame, textvariable=self.mutation_var, state="readonly")
            self.mutation_dropdown['values'] = ("Default", "Manual")
            self.mutation_dropdown.current(0)
            self.mutation_dropdown.grid(row=4, column=1, padx=5, pady=5)
            self.mutation_dropdown.bind("<<ComboboxSelected>>", self.update_mutation_inputs)

            self.mutation_inputs = {}
            self.create_mutation_inputs()

        if self.optimizer_var.get() == "Basin Hopping":
            for widget in self.input_frame.grid_slaves():
                if int(widget.grid_info()["row"]) > 3:
                    widget.grid_forget()

                  

    def create_mutation_inputs(self):
        labels = ["twist", "random displacement", "angular", "random step", "etching"]
        default_values = [0.3, 0.1, 0.3, 0.3, 0.1]
        for i, label in enumerate(labels):
            ttk.Label(self.input_frame, text=label.capitalize() + ":").grid(row=5+i, column=0, padx=5, pady=5)
            var = tk.DoubleVar(value=default_values[i])
            entry = ttk.Entry(self.input_frame, textvariable=var, width=10)
            entry.grid(row=5+i, column=1, padx=5, pady=5)
            self.mutation_inputs[label] = var

    def update_mutation_inputs(self, event):
        if self.mutation_var.get() == "Default":
            for label, var in self.mutation_inputs.items():
                var.set(OrderedDict([("twist", 0.3), ("random displacement", 0.1), ("angular", 0.3), ("random step", 0.3), ("etching", 0.1)])[label])
        else:
            for label, var in self.mutation_inputs.items():
                var.set(0.0)
    
    def run_optimizer(self):
        try:
            optimizer_choice = self.optimizer_var.get()
            calculator_choice = self.calculator_var.get()
            element = self.element_var.get()
            iterations = self.num_iter_var.get()
            conv_iterations = self.conv_iter_var.get()

            self.mutation = OrderedDict(
            [
            ("twist", self.mutation_inputs["twist"].get()),
            ("random displacement", self.mutation_inputs["random displacement"].get()),
            ("angular", self.mutation_inputs["angular"].get()),
            ("random step", self.mutation_inputs["random step"].get()),
            ("etching", self.mutation_inputs["etching"].get()),
            ])   

            self.log(f"Running {optimizer_choice} with {calculator_choice} Cluster='{element}', iterations={iterations}")
            calculator_ = self.get_calculator(calculator_choice)

            
            if optimizer_choice == "Genetic Algorithm":
                ga = GeneticAlgorithm(mutation=self.mutation, num_clusters=4, preserve=True, debug=True, calculator=calculator_)
                ga.run(element, iterations, conv_iterations)
                self.myatoms = ga.best_config

                if not os.path.exists("./gpaw"):
                    os.mkdir("./gpaw")

                with Trajectory("./gpaw/movie.traj", mode="w") as traj:  # type: ignore
                    for cluster in ga.configs:
                        cluster.center()
                        traj.write(cluster)

                self.log("Genetic Algorithm completed successfully.")
                self.plot_trajectory(ga.potentials)
                self.show_log()
                self.view_menu.add_separator()
                self.view_menu.add_command(label="Movie", command=self.show_movie)

                if calculator_choice == "GPAW":
                    if not os.path.exists("./gpaw"):
                        os.mkdir("./gpaw")

                    id = gpw(self.myatoms)
                    self.view_menu.add_command(label="Band Structure", command=self.show_band_structure(id))
                    self.show_log()        
                    

            elif optimizer_choice == "Basin Hopping":
                bh = BasinHoppingOptimizer(calculator=calculator_)
                bh.run(element, iterations)
                self.myatoms = bh.best_config

                if not os.path.exists("./gpaw"):
                    os.mkdir("./gpaw")

                with Trajectory("./gpaw/movie.traj", mode="w") as traj:  # type: ignore
                    for cluster in bh.configs:
                        cluster.center()
                        traj.write(cluster)

                self.log("Basin Hopping completed successfully.")
                self.plot_trajectory(bh.potentials)
                self.show_log()
                self.view_menu.add_separator()
                self.view_menu.add_command(label="Movie", command=self.show_movie)

                if calculator_choice == "GPAW":
                    id = gpw(self.myatoms)
                    self.view_menu.add_command(label="Band Structure", command=self.show_band_structure(id))
                    self.show_log()  

            else:
                messagebox.showerror("Error", "Invalid optimizer selection.")

        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_calculator(self, calculator_choice):
        if calculator_choice == "Lennard Jones":
            from ase.calculators.lj import LennardJones
            return LennardJones
        elif calculator_choice == "EMT":
            from ase.calculators.emt import EMT
            return EMT
        elif calculator_choice == "GPAW":
            from ase.calculators.lj import LennardJones
            return LennardJones
        elif calculator_choice == "EAM":
            from ase.calculators.eam import EAM
            return EAM
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



    def show_3d_model(self):
        self.hide_all_views()
        view(self.myatoms)
    
    def show_band_structure(self,id):
        self.hide_all_views()
        file_path = id
        data = np.loadtxt(file_path, skiprows=1)
        energy = data[:, 0]  # First column: energy in eV
        S_z = data[:, 3]     # Second column: S_z
        energy = np.delete(energy, 0)
        S_z = np.delete(S_z, 0)
        wavelength = 1240 / energy  # Convert energy (eV) to wavelength (nm)
        fig, ax = plt.subplots()
        ax.plot(wavelength, S_z, label='S_z')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('S_z')
        ax.set_title('Optical Photoabsorption Spectrum')
        ax.legend()
        ax.set_xlim(0, 1000)


        canvas = FigureCanvasTkAgg(fig, master=self.simulation_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def show_log(self):
        self.hide_all_views()
        if self.log_text:
            self.log_text.pack(pady=5)

    def show_movie(self):
        self.hide_all_views()
        self.movie_atoms = ase.io.read("./gpaw/movie.traj", index=":")
        view(self.movie_atoms)

    def hide_all_views(self):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        if self.log_text:
            self.log_text.pack_forget()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()
