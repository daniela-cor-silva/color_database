import tkinter as tk
from tkinter import messagebox
import json
import argparse
import os

# args
parser = argparse.ArgumentParser()
parser.add_argument("dataset_file")
parser.add_argument("--save_file", default="colors.json")
args = parser.parse_args()

filename = args.dataset_file
save_filename = args.save_file

# functions
def load_colors(filename):
    with open(filename, 'r') as file:
        return [tuple(map(int, line.strip().split(' '))) for line in file]

def save_classifications(filename, classifications, index):
    with open(filename, 'w') as file:
        json.dump({"index": index, "classifications": classifications}, file, indent=4)

def load_progress(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get("index", 0), data.get("classifications", {})
    return 0, {}

# GUI
class ColorClassifier:
    def __init__(self, root, colors):
        self.root = root
        self.colors = colors
        self.index, self.classifications = load_progress(save_filename)
        self.current_category = None
        
        self.canvas = tk.Canvas(root, width=200, height=200)
        self.canvas.pack()
        
        self.index_label = tk.Label(root, text=f"Index: {self.index+1}/{len(self.colors)}", font=("Arial", 12))
        self.index_label.pack()
        
        self.label = tk.Label(root, text="Classify the color:")
        self.label.pack()
        
        self.selected_label = tk.Label(root, text="Selected: None", font=("Arial", 12, "bold"))
        self.selected_label.pack()
        
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()
        
        self.categories = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Grey", "Black", "White"]
        self.buttons = {}
        
        for category in self.categories:
            btn = tk.Button(self.buttons_frame, text=category, command=lambda c=category: self.set_category(c))
            btn.pack(side=tk.LEFT)
            self.buttons[category] = btn
        
        self.confirm_button = tk.Button(root, text="Confirm", command=self.confirm_classification)
        self.confirm_button.pack()
        
        self.save_button = tk.Button(root, text="Save Progress", command=self.save_progress)
        self.save_button.pack()
        
        self.update_display()

    def update_display(self):
        if self.index < len(self.colors):
            rgb = self.colors[self.index]
            hex_color = "#%02x%02x%02x" % rgb
            self.canvas.create_rectangle(0, 0, 200, 200, fill=hex_color, outline=hex_color)
            self.selected_label.config(text="Selected: None")
            self.index_label.config(text=f"Index: {self.index+1}/{len(self.colors)}")
        else:
            messagebox.showinfo("Done", "All colors classified!")
            self.save_progress()
            self.root.quit()

    def set_category(self, category):
        self.current_category = category
        self.selected_label.config(text=f"Selected: {category}")
        for btn in self.buttons.values():
            btn.config(relief=tk.RAISED)
        self.buttons[category].config(relief=tk.SUNKEN)
    
    def confirm_classification(self):
        if self.current_category and self.index < len(self.colors):
            self.classifications[str(self.colors[self.index])] = self.current_category
            self.index += 1
            self.current_category = None
            self.update_display()
        else:
            messagebox.showwarning("Warning", "Please select a category before confirming.")
    
    def save_progress(self):
        save_classifications(save_filename, self.classifications, self.index)
        messagebox.showinfo("Saved", f"Progress has been saved to '{save_filename}'.")

# main
if __name__ == "__main__":
    colors = load_colors(filename)
    root = tk.Tk()
    root.title("Color Classifier")
    app = ColorClassifier(root, colors)
    root.mainloop()