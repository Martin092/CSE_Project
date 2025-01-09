import tkinter as tk
from PIL import Image, ImageTk

class StartMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Atomic Simulation Start Menu")
        self.root.geometry("800x600")  

        self.set_background("../../../Downloads/imagebg.webp")

        self.create_menu()

    def set_background(self, image_path):
            image = Image.open(image_path)
            self.background_image = ImageTk.PhotoImage(image)

            background_label = tk.Label(self.root, image=self.background_image)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)

    def create_menu(self):
        menu_frame = tk.Frame(self.root, bg="black", bd=0)
        menu_frame.place(relx=0.5, rely=0.5, anchor="center")

        button_font = ("Arial", 14, "bold")
        button_style = {"bg": "#ffffff", "fg": "#000000", "relief": "raised", "bd": 3}

        tk.Button(menu_frame, text="Start Simulation", font=button_font, **button_style,
                  command=self.start_simulation).pack(pady=10)
        tk.Button(menu_frame, text="Settings", font=button_font, **button_style,
                  command=self.settings).pack(pady=10)
        tk.Button(menu_frame, text="Quit", font=button_font, **button_style,
                  command=self.root.quit).pack(pady=10)

    def start_simulation(self):
        tk.messagebox.showinfo("Start Simulation", "Simulation starting...")

    def settings(self):
        tk.messagebox.showinfo("Settings", "Open settings menu.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StartMenu(root)
    root.mainloop()