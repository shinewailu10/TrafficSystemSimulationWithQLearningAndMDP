import tkinter as tk
from tkinter import messagebox
import os
import threading
import main 

class TrafficLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic AI Controller")
        self.root.geometry("400x400")
        self.root.configure(bg="#f0f0f0")

        title_label = tk.Label(root, text="🚦 AI Traffic Control 🚦", 
                               font=("Segoe UI", 20, "bold"), bg="#f0f0f0", fg="#333")
        title_label.pack(pady=20)

        desc_label = tk.Label(root, text="Select a mode to run the simulation:", 
                              font=("Segoe UI", 10), bg="#f0f0f0", fg="#666")
        desc_label.pack(pady=5)

        self.btn_train = tk.Button(root, text="🧠 Train New Agent (300 Episodes)", 
                                   command=self.train_new, 
                                   font=("Segoe UI", 11), bg="#ffcccc", width=30, height=2)
        self.btn_train.pack(pady=10)

        self.btn_run = tk.Button(root, text="😎 Run Smart Agent (Watch)", 
                                 command=self.run_smart, 
                                 font=("Segoe UI", 11), bg="#ccffcc", width=30, height=2)
        self.btn_run.pack(pady=10)

        self.btn_reset = tk.Button(root, text="🗑️ Delete Saved Brain", 
                                   command=self.delete_brain, 
                                   font=("Segoe UI", 10), bg="#e0e0e0", width=20)
        self.btn_reset.pack(pady=20)

        self.status_label = tk.Label(root, text="Status: Ready", 
                                     font=("Segoe UI", 10, "italic"), bg="#f0f0f0", fg="blue")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def delete_brain(self):
        if os.path.exists("traffic_brain.pkl"):
            try:
                os.remove("traffic_brain.pkl")
                self.status_label.config(text="Status: Brain deleted!")
                messagebox.showinfo("Success", "The AI brain has been reset.")
            except PermissionError:
                messagebox.showerror("Error", "Close the simulation first!")
        else:
            self.status_label.config(text="Status: No brain file found.")
            messagebox.showinfo("Info", "No brain file found to delete.")

    def run_simulation_thread(self):
        self.status_label.config(text="Status: Simulation Running...")
        
        self.btn_train.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        
        try:
            import importlib
            importlib.reload(main) 
            main.main() 
        except Exception as e:
            print(f"Error in simulation: {e}")
            messagebox.showerror("Simulation Error", f"An error occurred:\n{e}")
        
        self.status_label.config(text="Status: Finished!")
        self.btn_train.config(state=tk.NORMAL)
        self.btn_run.config(state=tk.NORMAL)
        self.btn_reset.config(state=tk.NORMAL)

    def train_new(self):
        if os.path.exists("traffic_brain.pkl"):
            try:
                os.remove("traffic_brain.pkl")
            except PermissionError:
                messagebox.showerror("Error", "Close the running simulation first!")
                return
        
        threading.Thread(target=self.run_simulation_thread, daemon=True).start()

    def run_smart(self):
        if not os.path.exists("traffic_brain.pkl"):
            messagebox.showwarning("Warning", "No trained brain found! \nPlease click 'Train New Agent' first.")
            return

        threading.Thread(target=self.run_simulation_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficLauncher(root)
    root.mainloop()