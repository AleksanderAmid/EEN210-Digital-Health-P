import tkinter as tk
from subprocess import Popen
"""
Update Notes(Tänker man kan skriva vad som updaterats i programmet här):

CLICK AND LABEL:
- The filename is now showed at the top of the window.
- A menu that ask for opening folder or file is now showed when the program is started.
- After Labeling the orginal file moves from readyToLabel to the trash and a copy of the labeled file is saved in the LabeledData folder.

READY TO LABEL:
- The filename is now showed at the top of the window.
- A menu that ask for opening folder or file is now showed when the program is started.

FURIER PLOT:
- Zooming and panning is now possible in the plot. (The menu is located in the bottom right corner of the plot)

PLOTTER (FIXED OFFSET):
-

PLOTTER (RAW):
-

"""
def open_click_and_label():
    Popen(['python', 'Program-ClickAndLabelV3.py'])
    root.destroy()

def open_ready_to_label():
    Popen(['python', 'Program-ReadyToLabelV17.py'])
    root.destroy()

def open_furier_plot():
    Popen(['python', 'Program-FurierPlot.py'])
    root.destroy()

def open_plotter_fixed_offset():
    Popen(['python', 'Program-Plotter(Fixed Offset).py'])
    root.destroy()

def open_plotter_raw():
    Popen(['python', 'Program-Plotter(RAW).py'])
    root.destroy()

def exit_app():
    root.destroy()

root = tk.Tk()
root.title("Choose Action")

title_label = tk.Label(root, text="Choose Program", font=("Helvetica", 16))
title_label.pack(pady=10)

btn_click_label = tk.Button(root, text="Click and Label", command=open_click_and_label, width=20)
btn_click_label.pack(pady=10)

btn_ready_label = tk.Button(root, text="Ready to Label", command=open_ready_to_label, width=20)
btn_ready_label.pack(pady=10)

btn_furier_plot = tk.Button(root, text="Furier Plot", command=open_furier_plot, width=20)
btn_furier_plot.pack(pady=10)

btn_plotter_fixed_offset = tk.Button(root, text="Plotter (Fixed Offset)", command=open_plotter_fixed_offset, width=20)
btn_plotter_fixed_offset.pack(pady=10)

btn_plotter_raw = tk.Button(root, text="Plotter (RAW)", command=open_plotter_raw, width=20)
btn_plotter_raw.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", command=exit_app, width=20)
btn_exit.pack(pady=10)

root.mainloop()