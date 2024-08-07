import tkinter as tk
from tkinter import scrolledtext, ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Create tkinter window
window = tk.Tk()
window.title("Plot Switcher")

# Read the CSV file
df = pd.read_csv(r"D:\BCA MINI PROJECT\dataset.csv", encoding='utf-8')
df['Joining_Date'] = pd.to_datetime(df['Joining_Date'], dayfirst=True)
df['Year'] = df['Joining_Date'].dt.year

def convert_experience_to_days(experience_str):
    years, months, days = 0, 0, 0
    experience_str = experience_str.replace(',', '')  # Remove any commas

    if 'Years' in experience_str:
        years = int(experience_str.split('Years')[0].strip())
        experience_str = experience_str.split('Years')[1].strip()
    if 'Months' in experience_str:
        months_part = experience_str.split('Months')[0].strip()
        if months_part.isdigit():
            months = int(months_part)
        experience_str = experience_str.split('Months')[1].strip()
    if 'Days' in experience_str:
        days = int(experience_str.split('Days')[0].strip())
    total_days = years * 365 + months * 30 + days
    return total_days

df['Year_of_experience'] = df['Year_of_experience'].apply(convert_experience_to_days)

# Standardize gender values
df['gender'] = df['gender'].replace({'m': 'Male', 'f': 'Female', 'M': 'Male', 'F': 'Female'})

# Read and display an image
image_path = r"D:\BCA MINI PROJECT\IMG.jpg"
image = Image.open(image_path)
image = image.resize((1500, 600), resample=Image.LANCZOS)
photo = ImageTk.PhotoImage(image)

# Display the image
image_label = tk.Label(window, image=photo)
image_label.pack()

# Function to redirect stdout to the Tkinter window
class StdoutRedirector:
    def _init_(self):
        pass

    def write(self, text):
        info_text.insert(tk.END, text)
        info_text.see(tk.END)  # Auto-scroll to the bottom

# Function to fill missing values
def fill_missing_values():
    global df
    df = df.ffill()  # Forward fill missing values
    print("Missing values filled using forward fill method.")
    # Update displayed dataset information
    show_dataset_info()

# Function to display dataset information
def show_dataset_info():
    info_window = tk.Toplevel(window)
    info_window.title("Dataset Information")
    info_window.geometry("500x500")
    # Create scrolled text widget
    global info_text
    info_text = scrolledtext.ScrolledText(info_window, width=80, height=20)
    info_text.pack()

    # Redirect stdout to the scrolled text widget
    sys.stdout = StdoutRedirector()
    
    # Print DataFrame head and info
    print("DataFrame head:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info())
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Button to fill missing values
    fill_button = tk.Button(info_window, text="Fill Missing Values", command=fill_missing_values, bg='lightblue', fg='black', height=4, width=20)
    fill_button.pack(pady=10)

# Function to show correlation matrix
def show_correlation_matrix():
    def plot_correlation_matrix():
        selected_year = int(year_combobox.get())
        selected_columns = ['age', 'Year_of_experience', 'avg_training_score', 'previous_year_rating', 'KPIs_met_more_than_80']
        filtered_df = df[df['Year'] == selected_year]
        numeric_df = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
        correlation_matrix = numeric_df.corr()

        # Exclude self-correlations
        np.fill_diagonal(correlation_matrix.values, np.nan)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=0.5, cbar=False)
        plt.title(f'Correlation Matrix ({selected_year})')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    corr_window = tk.Toplevel(window)
    corr_window.title("Correlation Matrix")

    # Combobox to select year
    years = sorted(df['Year'].unique())
    year_combobox = ttk.Combobox(corr_window, values=years)
    year_combobox.set(min(years))  # Set default value to the earliest year
    year_combobox.pack()

    # Button to plot selected year
    plot_button = tk.Button(corr_window, text="Show Correlation Matrix", command=plot_correlation_matrix, height=2, width=20)
    plot_button.pack(padx=30, pady=30)

def show_bar_plot():
    def plot_performance_by_gender():
        metrics = ['avg_training_score', 'Year_of_experience']
        genders = df['gender'].unique()
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        xpos = np.arange(len(metrics))
        ypos = np.arange(len(genders))
        xposM, yposM = np.meshgrid(xpos, ypos, indexing="ij")
        xposM = xposM.flatten()
        yposM = yposM.flatten()
        zposM = np.zeros_like(xposM)

        dx = dy = 0.5
        dz = []

        colors = {'Male': 'blue', 'Female': 'pink'}  # Define colors for each gender

        for metric in metrics:
            for gender in genders:
                value = df[df['gender'] == gender][metric].mean()
                dz.append(value)

        dz = np.array(dz)

        for i in range(len(dz)):
            ax.bar3d(xposM[i], yposM[i], zposM[i], dx, dy, dz[i], shade=True, color=colors[genders[yposM[i]]])

        # Set ticks and labels for x-axis (metrics)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')

        # Set ticks and labels for y-axis (genders)
        ax.set_yticks(np.arange(len(genders)))
        ax.set_yticklabels(genders)

        # Set labels for z-axis (average value)
        ax.set_zlabel('Average Value')

        # Set axis labels
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Gender')

        plt.title('Performance Metrics by Gender (3D Bar Plot)')

        # Create a custom legend
        legend_elements = [plt.Line2D([0], [0], color=colors[gender], lw=4, label=gender) for gender in genders]
        ax.legend(handles=legend_elements, loc='best')

        plt.show()

    # Directly plot performance metrics by gender
    plot_performance_by_gender()


# Function to show box plot for overall performance trend
def show_overall_performance():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y='avg_training_score')
    plt.title('Overall Performance Trend (Box Plot)')
    plt.ylabel('Average Training Score')
    plt.grid(True)
    
    # Annotate the plot with the employee with the highest performance
    max_score = df['avg_training_score'].max()
    max_employee = df.loc[df['avg_training_score'].idxmax()]  # Employee with the highest performance
    plt.annotate(f'Highest Performance: {max_employee["employee_id"]} - {max_score}', 
                 xy=(0.5, max_score), xytext=(0.5, max_score + 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, ha='center')
    
    # Annotate the plot with the employee with the lowest performance
    min_score = df['avg_training_score'].min()
    min_employee = df.loc[df['avg_training_score'].idxmin()]  # Employee with the lowest performance
    plt.annotate(f'Lowest Performance: {min_employee["employee_id"]} - {min_score}', 
                 xy=(0.5, min_score), xytext=(0.5, min_score - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, ha='center')
    
    plt.show()

# Function to exit the application
def exit_application():
    window.quit()
    window.destroy()

# Button to show dataset information
info_button = tk.Button(window, text="Dataset Info", command=show_dataset_info, bg='lightblue', fg='black', height=4, width=20)
info_button.pack(side=tk.LEFT, padx=50, pady=60)

# Button to show correlation matrix
corr_button = tk.Button(window, text="Show Correlation Matrix", command=show_correlation_matrix, bg='lightblue', fg='black', height=4, width=20)
corr_button.pack(side=tk.LEFT, padx=90, pady=10)

# Button to show bar plot
bar_button = tk.Button(window, text="Show Bar Plot", command=show_bar_plot, bg='lightblue', fg='black', height=4, width=20)
bar_button.pack(side=tk.LEFT, padx=90, pady=40)

# Button to show overall performance trend
overall_button = tk.Button(window, text="Overall Performance Trend", command=show_overall_performance, bg='lightblue', fg='black', height=4, width=25)
overall_button.pack(side=tk.LEFT, padx=50, pady=40)

# Button to exit the application
exit_button = tk.Button(window, text="Exit", command=exit_application, bg='lightcoral', fg='black', height=4, width=20)
exit_button.pack(side=tk.LEFT, padx=100, pady=30)

# Run the tkinter event loop
window.mainloop()
