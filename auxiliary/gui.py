"""GUI module"""

import sys
import os
from collections import OrderedDict
from typing import Any, List

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageTk
import ase
from ase import Atoms
from ase.io import Trajectory
from ase.visualize import view
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM

sys.path.append("./")
matplotlib.use("Agg")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from src.basin_hopping_optimizer import (
    BasinHoppingOptimizer,
    OperatorSequencing,
)  # pylint: disable=C0413
from src.global_optimizer import GlobalOptimizer  # pylint: disable=C0413
from auxiliary.gpw import gpw  # pylint: disable=C0413


class OptimizerGUI:
    """
    GUI class structure
    """

    def __init__(self, r: tk.Tk):
        self.root = r
        self.root.title("Atomic Materials Structure Optimizer")
        self.root.geometry("800x600")
        self.start_frame = tk.Frame(self.root)
        self.start_frame.pack(expand=True, fill="both")
        self.create_start()
        self.simulation_frame: tk.Frame
        self.background_image: ImageTk.PhotoImage
        self.optimizer_var: tk.StringVar
        self.calculator_var: tk.StringVar
        self.view_menu: tk.Menu
        self.element_var: tk.StringVar
        self.num_iter_var: tk.IntVar
        self.conv_iter_var: tk.IntVar
        self.parameter_frame: tk.Frame
        self.num_cluster: tk.IntVar
        self.num_select: tk.IntVar
        self.preserve: tk.BooleanVar
        self.mutation_inputs: dict[Any, Any] = {}
        self.displacement_length: tk.DoubleVar
        self.operators: dict[Any, Any] = {}
        self.static: tk.BooleanVar
        self.reject_lab: ttk.Label
        self.max_rejects: tk.IntVar
        self.rejects_entry: ttk.Entry
        self.myatoms: Atoms
        self.settings_button: ttk.Button
        self.show_params: bool = False
        self.log: str = ""
        self.log_field: tk.Label

    def create_start(self) -> None:
        """
        Generates the initial start frame.
        """
        image = Image.open("static/imagebg.webp")
        self.background_image = ImageTk.PhotoImage(image)

        background_label = tk.Label(self.start_frame, image=self.background_image)  # type: ignore
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

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
            "activeforeground": "#000000",
        }

        tk.Button(
            menu_frame,
            text="Start Simulation",
            **button_style,  # type: ignore
            command=self.start_simulation,
        ).pack(pady=10)
        tk.Button(menu_frame, text="Quit", **button_style, command=self.root.quit).pack(  # type: ignore
            pady=10
        )

    def start_simulation(self) -> None:
        """
        Generates the simulation frame.
        """
        self.start_frame.destroy()
        self.create_menu()
        self.simulation_frame = tk.Frame(self.root)
        self.simulation_frame.pack(expand=True, padx=20, pady=20)

        ttk.Label(
            self.simulation_frame,
            text="Atomic Materials Structure Optimizer",
            font=("Arial", 16),
        ).grid(row=0, column=0, columnspan=3, pady=5, padx=5)

        ttk.Label(self.simulation_frame, text="Select Optimization Method:").grid(
            row=1, column=0, padx=5, pady=5
        )
        self.optimizer_var = tk.StringVar()
        optimizer_dropdown = ttk.Combobox(
            self.simulation_frame, textvariable=self.optimizer_var, state="readonly"
        )
        optimizer_dropdown["values"] = (" ", "Genetic Algorithm", "Basin Hopping")
        optimizer_dropdown.current(0)
        optimizer_dropdown.grid(row=1, column=1, padx=5, pady=5)

        image = Image.open("static/settings.jpeg")
        settings_icon = ImageTk.PhotoImage(image.resize((25, 25)))
        self.settings_button = ttk.Button(
            self.simulation_frame,
            image=settings_icon,  # type: ignore
            command=self.setting,
        )
        self.settings_button.image = settings_icon  # type: ignore
        self.settings_button.grid(row=1, column=2, padx=5, pady=5)
        self.settings_button.grid_forget()

        optimizer_dropdown.bind("<<ComboboxSelected>>", self.toggle_settings_button)

        ttk.Label(self.simulation_frame, text="Select Interaction/Calculator:").grid(
            row=2, column=0, padx=5, pady=5
        )
        self.calculator_var = tk.StringVar()
        calculator_dropdown = ttk.Combobox(
            self.simulation_frame, textvariable=self.calculator_var, state="readonly"
        )
        calculator_dropdown["values"] = (
            " ",
            "Lennard Jones",
            "EMT",
            "EAM",
            "GPAW",
        )
        calculator_dropdown.current(0)
        calculator_dropdown.grid(row=2, column=1, padx=5, pady=5)

        # TODO: Add Local Optimizers (BFGS, FIRE for sure, maybe some other?)

        self.create_input_fields()

        ttk.Button(
            self.simulation_frame, text="Run Optimizer", command=self.run_optimizer
        ).grid(row=7, column=0, columnspan=3, padx=5, pady=5)

        self.log_field = tk.Label(self.simulation_frame)

    def toggle_settings_button(self, _: Any) -> None:
        """
        Toggles settings button when an optimizer is selected.
        """
        selected_option = self.optimizer_var.get()

        if selected_option != " ":
            self.settings_button.grid(row=1, column=2, padx=5, pady=5)
        else:
            self.settings_button.grid_forget()

        if self.show_params:
            self.params()

    def create_menu(self) -> None:
        """
        Generates filling menu.
        """
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        self.view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_command(label="Graph", command=self.show_plot)
        self.view_menu.add_command(
            label="3D Interactive Model", command=self.show_3d_model
        )
        self.view_menu.add_command(label="Movie", command=self.show_movie)
        self.view_menu.add_command(
            label="Band Structure", command=self.show_band_structure
        )

    def create_input_fields(self) -> None:
        """
        Generates general input fields such as cluster atomic type,
        max number of iterations and number of convergence iterations.
        """
        ttk.Label(
            self.simulation_frame,
            text="Atoms (e.g., C2H4 for 2 carbon and 4 hydrogen atoms):",
        ).grid(row=3, column=0, padx=5, pady=5)
        self.element_var = tk.StringVar(value="C2H4")
        ttk.Entry(self.simulation_frame, textvariable=self.element_var, width=10).grid(
            row=3, column=1, padx=5, pady=5
        )

        ttk.Label(
            self.simulation_frame, text="Max Number of Iterations for Execution:"
        ).grid(row=4, column=0, padx=5, pady=5)
        self.num_iter_var = tk.IntVar(value=50)
        ttk.Entry(self.simulation_frame, textvariable=self.num_iter_var, width=10).grid(
            row=4, column=1, padx=5, pady=5
        )

        ttk.Label(
            self.simulation_frame,
            text="Number of Iterations Considered for Convergence:",
        ).grid(row=5, column=0, padx=5, pady=5)
        self.conv_iter_var = tk.IntVar(value=10)
        ttk.Entry(
            self.simulation_frame, textvariable=self.conv_iter_var, width=10
        ).grid(row=5, column=1, padx=5, pady=5)

        self.parameter_frame = tk.Frame(self.simulation_frame)
        self.parameter_frame.grid(row=6, column=0, padx=5, pady=5, columnspan=3)
        self.create_ga_inputs()
        self.parameter_frame.destroy()
        self.parameter_frame = tk.Frame(self.simulation_frame)
        self.parameter_frame.grid(row=6, column=0, padx=5, pady=5, columnspan=3)
        self.create_bh_inputs()
        self.parameter_frame.destroy()
        self.parameter_frame = tk.Frame(self.simulation_frame)
        self.parameter_frame.grid(row=6, column=0, padx=5, pady=5, columnspan=3)

    def setting(self) -> None:
        """
        Changes whether hyperparameter fields should be shown and refreshes the frame.
        """
        self.show_params = not self.show_params
        self.params()

    def params(self) -> None:
        """
        Generates relevant hyperparameter fields.
        """
        if self.optimizer_var.get() != " " and self.show_params:
            if self.optimizer_var.get() == "Genetic Algorithm":
                self.parameter_frame.destroy()
                self.parameter_frame = tk.Frame(self.simulation_frame)
                self.parameter_frame.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
                self.create_ga_inputs()
            elif self.optimizer_var.get() == "Basin Hopping":
                self.parameter_frame.destroy()
                self.parameter_frame = tk.Frame(self.simulation_frame)
                self.parameter_frame.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
                self.create_bh_inputs()
            else:
                self.parameter_frame.grid_forget()
        else:
            if hasattr(self, "parameter_frame"):
                self.parameter_frame.grid_forget()

    def create_bh_inputs(self) -> None:
        """
        Generates input fields for Basin Hopping hyperparameters.
        """
        ttk.Label(self.parameter_frame, text="Operators").grid(
            row=0, column=0, rowspan=3, padx=5, pady=5
        )
        ttk.Label(self.parameter_frame, text="Steps").grid(
            row=0, column=3, rowspan=3, padx=5, pady=5
        )
        labels = ["random step", "twist", "angular"]
        default_values = [5, 5, 5]
        for i, label in enumerate(labels):
            ttk.Label(self.parameter_frame, text=label.capitalize() + ":").grid(
                row=i, column=1, padx=5, pady=5
            )
            var = tk.IntVar(value=default_values[i])
            ttk.Entry(self.parameter_frame, textvariable=var, width=10).grid(
                row=i, column=2, padx=5, pady=5
            )
            self.operators[label] = var
        ttk.Label(self.parameter_frame, text="Static:").grid(
            row=3, column=0, padx=5, pady=5, columnspan=2
        )
        self.static = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.parameter_frame,
            variable=self.static,
            width=10,
            command=self.toggle_rejects,
        ).grid(row=3, column=2, padx=5, pady=5, columnspan=2)
        self.reject_lab = ttk.Label(self.parameter_frame, text="Max rejects:")
        self.max_rejects = tk.IntVar(value=5)
        self.rejects_entry = ttk.Entry(
            self.parameter_frame, textvariable=self.max_rejects, width=10
        )

    def toggle_rejects(self) -> None:
        """
        Toggles the max_reject BH hyperparameter when static hyperparameter is false.
        """
        if self.static.get():
            self.rejects_entry.grid_forget()
            self.reject_lab.grid_forget()
        else:
            self.reject_lab.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
            self.rejects_entry.grid(row=4, column=2, padx=5, pady=5, columnspan=2)

    def create_ga_inputs(self) -> None:
        """
        Generates input fields for Genetic Algorithm hyperparameters.
        """
        ttk.Label(
            self.parameter_frame, text="Number of clusters for optimization:"
        ).grid(row=0, column=0, padx=5, pady=5, columnspan=2)
        self.num_cluster = tk.IntVar(value=8)
        ttk.Entry(self.parameter_frame, textvariable=self.num_cluster, width=10).grid(
            row=0, column=2, padx=5, pady=5, columnspan=2
        )
        ttk.Label(self.parameter_frame, text="Number of clusters for selection:").grid(
            row=1, column=0, padx=5, pady=5, columnspan=2
        )
        self.num_select = tk.IntVar(value=max(2, int(self.num_cluster.get() / 2)))
        ttk.Entry(self.parameter_frame, textvariable=self.num_select, width=10).grid(
            row=1, column=2, padx=5, pady=5, columnspan=2
        )
        ttk.Label(self.parameter_frame, text="Preserve selected clusters:").grid(
            row=2, column=0, padx=5, pady=5, columnspan=2
        )
        self.preserve = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.parameter_frame, variable=self.preserve, width=10).grid(
            row=2, column=2, padx=5, pady=5, columnspan=2
        )
        ttk.Label(self.parameter_frame, text="Mutations").grid(
            row=3, column=0, rowspan=5, padx=5, pady=5
        )
        labels = ["twist", "angular", "random step", "etching", "random displacement"]
        default_values = [0.3, 0.3, 0.3, 0.1, 0.1]
        for i, label in enumerate(labels):
            ttk.Label(self.parameter_frame, text=label.capitalize() + ":").grid(
                row=3 + i, column=1, padx=5, pady=5
            )
            var = tk.DoubleVar(value=default_values[i])
            ttk.Entry(self.parameter_frame, textvariable=var, width=10).grid(
                row=3 + i, column=2, padx=5, pady=5
            )
            self.mutation_inputs[label] = var
        ttk.Label(self.parameter_frame, text="Probabilities").grid(
            row=3, column=3, rowspan=4, padx=5, pady=5
        )
        ttk.Label(self.parameter_frame, text="Length of displacement vector:").grid(
            row=8, column=0, padx=5, pady=5, columnspan=2
        )
        self.displacement_length = tk.DoubleVar(value=0.1)
        ttk.Entry(self.parameter_frame, textvariable=self.displacement_length).grid(
            row=8, column=2, padx=5, pady=5, columnspan=2
        )

    def run_optimizer(self) -> None:
        """
        Executes the global optimizer.
        """
        try:
            self.log = ""
            self.log_field.grid(row=8, column=0, columnspan=3, padx=5, pady=5)
            self.log_field.config(text=self.log)

            optimizer_choice = self.optimizer_var.get()
            calculator_choice = self.calculator_var.get()
            element = self.element_var.get()
            iterations = self.num_iter_var.get()
            conv_iterations = self.conv_iter_var.get()

            if self.optimizer_var.get() == "Genetic Algorithm":
                mutation = OrderedDict(
                    [
                        ("twist", self.mutation_inputs["twist"].get()),
                        (
                            "random displacement",
                            self.mutation_inputs["random displacement"].get(),
                        ),
                        ("angular", self.mutation_inputs["angular"].get()),
                        ("random step", self.mutation_inputs["random step"].get()),
                        ("etching", self.mutation_inputs["etching"].get()),
                    ]
                )
                num_clus = self.num_cluster.get()
                num_select = self.num_select.get()
                pres = self.preserve.get()

            elif self.optimizer_var.get() == "Basin Hopping":
                operators = OrderedDict(
                    [
                        ("random step", self.operators["random step"].get()),
                        ("twist", self.operators["twist"].get()),
                        ("angular", self.operators["angular"].get()),
                    ]
                )
                static = self.static.get()
                max_rejects = self.max_rejects.get()

            calculator_ = self.get_calculator(calculator_choice)

            optimizer: GlobalOptimizer

            if optimizer_choice == "Genetic Algorithm":
                optimizer = GeneticAlgorithm(
                    mutation=mutation,  # pylint: disable=E0606
                    num_clusters=num_clus,  # pylint: disable=E0606
                    num_selection=num_select,  # pylint: disable=E0606
                    preserve=pres,  # pylint: disable=E0606
                    calculator=calculator_,
                )

            elif optimizer_choice == "Basin Hopping":
                optimizer = BasinHoppingOptimizer(
                    operator_sequencing=OperatorSequencing(
                        operators=operators,  # pylint: disable=E0601
                        static=static,  # pylint: disable=E0601
                        max_rejects=max_rejects,  # pylint: disable=E0601
                    ),
                    calculator=calculator_,
                )

            self.log += f"Executing {optimizer_choice}..."
            self.log_field.config(text=self.log)

            optimizer.run(element, iterations, conv_iterations)

            self.log += "\nExecution finished."
            self.log_field.config(text=self.log)

            self.myatoms = optimizer.best_config

            if not os.path.exists("./gpaw"):
                os.mkdir("./gpaw")

            with Trajectory("./gpaw/movie.traj", mode="w") as traj:  # type: ignore
                for cluster in optimizer.configs:
                    cluster.center()  # type: ignore
                    traj.write(cluster)  # pylint: disable=E1101

            self.plot_graph(optimizer.potentials)

            if calculator_choice == "GPAW":
                gpw(self.myatoms)  # pylint: disable=W0622
                self.plot_band_structure()

        except Exception as e:  # pylint: disable=W0718
            messagebox.showerror("Error", f"An error occurred: {e}")

    def get_calculator(self, calculator_choice: str) -> Any:
        """
        Returns an object of the chosen calculator.
        :param calculator_choice: string name of the calculator.
        """
        if calculator_choice == "Lennard Jones":
            return LennardJones
        if calculator_choice == "EMT":
            return EMT
        if calculator_choice == "GPAW":
            return LennardJones
        if calculator_choice == "EAM":
            return EAM
        raise ValueError("Unsupported calculator.")

    def plot_graph(self, potentials: List[float]) -> None:
        """
        Saves potentials plot.
        :param potentials: List of the potentials to be plotted.
        """
        plt.plot(potentials)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ticks = list(plt.gca().get_xticks())[1:-1]
        if ticks[-1] != len(potentials) - 1:
            ticks.append(len(potentials) - 1)
        plt.gca().set_xticks(ticks)
        plt.title(f"Execution on {self.element_var.get()}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.tight_layout()
        plt.savefig("gpaw/plot.png")

    def show_plot(self) -> None:
        """
        Displays potentials plot.
        """
        image = Image.open("./gpaw/plot.png")
        image.show()

    def show_3d_model(self) -> None:
        """
        Displays 3D model of the optimized cluster.
        """
        self.myatoms.center()  # type: ignore
        view(self.myatoms)  # type: ignore

    def plot_band_structure(self) -> None:  # pylint: disable=W0622
        """
        Saves band structure graph.
        """
        data = np.loadtxt("./gpaw/be_spectrum_z2.dat", skiprows=1)
        energy = data[:, 0]  # First column: energy in eV
        s_z = data[:, 3]  # Second column: S_z
        energy = np.delete(energy, 0)
        s_z = np.delete(s_z, 0)
        wavelength = 1240 / energy  # Convert energy (eV) to wavelength (nm)
        fig, ax = plt.subplots()
        ax.plot(wavelength, s_z, label="S_z")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("S_z")
        ax.set_title("Optical Photoabsorption Spectrum")
        ax.legend()
        ax.set_xlim(0, 1000)
        fig.savefig("./gpaw/spectrum.png")

    def show_band_structure(self) -> None:
        """
        Displays band structure graph.
        """
        image = Image.open("./gpaw/spectrum.png")
        image.show()

    def show_movie(self) -> None:
        """
        Displays trajectory file.
        """
        movie_atoms = ase.io.read("./gpaw/movie.traj", index=":")
        view(movie_atoms)  # type: ignore


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()
