import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("To-Do List")

def add_task():
    task = task_entry.get()
    if task:
        tasks_listbox.insert(tk.END, task)
        task_entry.delete(0, tk.END)
def delete_task():
        selected_task_index = tasks_listbox.curselection()[0]
        tasks_listbox.delete(selected_task_index)

def mark_done():
        selected_task_index = tasks_listbox.curselection()[0]
        task = tasks_listbox.get(selected_task_index)
        tasks_listbox.delete(selected_task_index)
        tasks_listbox.insert(tk.END, f"{task} (Done)")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

task_entry = ttk.Entry(frame, width=40)
task_entry.grid(row=0, column=0, padx=5, pady=5)

add_button = ttk.Button(frame, text="Add Task", command=add_task)
add_button.grid(row=0, column=1, padx=5, pady=5)

tasks_listbox = tk.Listbox(frame, height=15, width=50)
tasks_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tasks_listbox.yview)
scrollbar.grid(row=1, column=2, sticky=(tk.N, tk.S))
tasks_listbox.config(yscrollcommand=scrollbar.set)

delete_button = ttk.Button(frame, text="Delete Task", command=delete_task)
delete_button.grid(row=2, column=0, padx=5, pady=5)

done_button = ttk.Button(frame, text="Mark Done", command=mark_done)
done_button.grid(row=2, column=1, padx=5, pady=5)

root.mainloop() 