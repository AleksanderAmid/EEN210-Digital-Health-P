from Offset_fixing_samefile import fix_offset
from Data_ClickAndWriteLabelsV2 import axel_fix, save_data, make_onselect, spanselector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from matplotlib.widgets import SpanSelector, Button

df = None

def main():
    global df
    #filepath
    file = 'data3/gait_sit_squat_jump_fall__obstacle_data_20250129_115458.csv'

    #fix data offset
    offset_file = fix_offset(file)
    df = pd.read_csv(offset_file)
    #print plotaxis
    ax_acc, ax_gyro, time = axel_fix(df)
    spanselector(ax_acc, ax_gyro, time, df)

if __name__ == "__main__":
    main()