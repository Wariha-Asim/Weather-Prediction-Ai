import pandas as pd  # Data manipulation aur analysis ke liye
import tkinter as tk  # GUI (Graphical User Interface) banane ke liye
from datetime import datetime, timedelta  # Date aur time handle karne ke liye
from sklearn.linear_model import LinearRegression  # Linear regression model predictions ke liye
from sklearn.metrics import mean_absolute_error, r2_score  # Model ki performance evaluate karne ke liye
import matplotlib.pyplot as plt  # Graphs aur visualizations banane ke liye
import tkinter.messagebox  # Pop-up message boxes banane ke liye
from PIL import Image, ImageTk  # Images handle aur display karne ke liye


# Global variable jo predicted data ko store karega
predicted_data = None
historical_data = None  # Global variable jo historical data ko store karega

# Function jo historical weather data ko CSV se fetch karega
def fetch_historical_data(city, file_path="../cleaned_daily_weather_data.csv"):
    try:
        data = pd.read_csv(file_path)  # CSV file se data read kare
        city_data = data[data['city'].str.lower() == city.lower()].copy()  # City ka data filter kare
        if city_data.empty:
            return pd.DataFrame()  # Agar city nahi milti to khali DataFrame return kare
        city_data['date'] = pd.to_datetime(city_data['date'], format='%d-%m-%Y')  # Date ko datetime format mein convert kare
        return city_data[['date', 'tavg', 'tmin', 'tmax', 'wspd', 'wdir', 'pres']]  # Required columns return kare
    except FileNotFoundError:
        print(f"File not found: {file_path}")  # Agar file nahi milti to error print kare
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")  # Kisi aur error par message print kare
        return pd.DataFrame()

# Function jo future temperatures ko predict karega
def predict_temperature(historical_data):
    if historical_data.empty:
        return pd.DataFrame(columns=['date', 'predicted_temperature']), 0, 0  # Agar data khali hai to khali DataFrame return kare

    # Date ko ordinal mein convert kare regression ke liye
    historical_data['date_ordinal'] = historical_data['date'].apply(lambda x: x.toordinal())
    X = historical_data[['date_ordinal']].values  #independent variable
    y = historical_data['tavg']  # dependent variable
    
    # Linear Regression model ko train kare
    model = LinearRegression()
    model.fit(X, y)

    # Agle 7 dinon ke liye prediction generate kare
    future_dates = pd.date_range(datetime.now() + timedelta(days=1), periods=7)
    # Future dates ko ordinal numbers (days since Year 1) me convert karein ML model ke liye
    # .map(lambda x: x.toordinal()) har date ko integer me badalta hai
    # .reshape(-1, 1) array ko 2D format me change karta hai jo ML model ko chahiye hota hai
    # .values result ko NumPy array me convert karta hai

    future_dates_ordinal = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    predicted_temps = model.predict(future_dates_ordinal)

    # Errors calculate kare (training set par)
    mae = mean_absolute_error(y, model.predict(X))
    r2 = r2_score(y, model.predict(X))

    # Predictions ke liye DataFrame banaye
    forecast = pd.DataFrame({'date': future_dates, 'predicted_temperature': predicted_temps})
    return forecast, mae, r2

def show_predictions():
    global predicted_data, historical_data  # Global variable ko use kare predictions store karne ke liye
    city = city_entry.get()  # City ka naam entry se le
    date_input = date_entry.get()  # Date ka input le

    # Date input ki validation kare
    if date_input:
        try:
            input_date = datetime.strptime(date_input, "%d-%m-%Y")  # Date ko parse kare
            # Check kare agar input date agle 7 dinon mein hai
            if input_date < datetime.now() or input_date > datetime.now() + timedelta(days=7):
                tk.messagebox.showerror("Error", "Please enter a date within the next 7 days.")  # Error message agar date valid na ho
                return
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid date format. Please use date-month-year.")  # Agar format galat ho to error
            return

    # Historical data ko fetch kare jo selected city ke liye hai
    historical_data = fetch_historical_data(city)  # Historical data fetch kare
    
    # Agar city ka data nahi milta to error dikhaye
    if historical_data.empty:
        tk.messagebox.showerror("Error", "No data found for the entered city.")  # City ka data nahi mila
        return

    # Predictions lene ke liye model ka use kare
    predicted_data, mae, r2 = predict_temperature(historical_data)  # Predictions le

    # Naya window banaye jisme predictions dikhaye
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Temperature Prediction")
    prediction_window.geometry("500x600")
    prediction_window.configure(bg='#00AFFF')  # Window ka background color set kare

    # Title label banaye
    tk.Label(prediction_window, text="Temperature Prediction For Next 7 Days", font=("Times New Roman ", 16, "bold"), bg='white').pack(pady=10)

    # Grid ke liye frame banaye
    grid_frame = tk.Frame(prediction_window, bg='white')
    grid_frame.pack(pady=10)

    # Predictions ko grid format mein dikhaye
    tk.Label(grid_frame, text="Date", font=("Times New Roman", 12, "bold"), bg='white').grid(row=0, column=0, padx=10, pady=5)  # Date ka header
    tk.Label(grid_frame, text="Day", font=("Times New Roman", 12, "bold"), bg='white').grid(row=0, column=1, padx=10, pady=5)  # Day ka header
    tk.Label(grid_frame, text="Predicted Temperature ( °C)", font=("Times New Roman", 12, "bold"), bg='white').grid(row=0, column=2, padx=10, pady=5)  # Temperature ka header

    # Har prediction ko grid mein dikhaye
    for index, row in predicted_data.iterrows():  # Data ko row by row iterate kare
        day_name = row['date'].day_name()  # Hafte ka din le
        # Predictions ko grid mein dikhaye
        tk.Label(grid_frame, text=row['date'].date(), font=("Times New Roman", 12), bg='white').grid(row=index + 1, column=0, padx=10, pady=5)  # Date ko display kare
        tk.Label(grid_frame, text=day_name, font=("Times New Roman", 12), bg='white').grid(row=index + 1, column=1, padx=10, pady=5)  # Day name ko display kare
        tk.Label(grid_frame, text=f"{row['predicted_temperature']:.2f}", font=("Times New Roman", 12), bg='white').grid(row=index + 1, column=2, padx=10, pady=5)  # Predicted temperature ko display kare

    # Mean Absolute Error aur R-squared dikhaye
    tk.Label(prediction_window, text="Mean Absolute Error:", font=("Times New Roman", 12, "bold"), bg='white').pack(pady=5)  # MAE label
    tk.Label(prediction_window, text=f"{mae:.2f}°C", font=("Times New Roman", 12), bg='white').pack(pady=5)  # MAE value dikhaye
    tk.Label(prediction_window, text="R-squared:", font=("Times New Roman", 12, "bold"), bg='white').pack(pady=5)  # R-squared label
    tk.Label(prediction_window, text=f"{r2:.2f}", font=("Times New Roman", 12), bg='white').pack(pady=5)  # R-squared value dikhaye


def plot_historical_data(historical_data):
    plt.figure(figsize=(10, 7))  # Graph ka size set karte hain (width=10, height=7)
    plt.plot(historical_data['date'], historical_data['tavg'], label='Historical Avg Temperature', color='blue', marker='o')  
    # Historical average temperature ko plot karte hain (dates x-axis par aur temperature y-axis par)
    plt.xlabel('Years')  # X-axis ka label 'Years' set karte hain
    plt.ylabel('Temperature (°C)')  # Y-axis ka label 'Temperature (°C)' set karte hain
    plt.grid(True)  # Graph me grid lines dikhate hain
    plt.tight_layout()  # Graph ke layout ko adjust karte hain taake sab kuch sahi se dikhe
    plt.title('Historical Temperature')  # Graph ka title set karte hain
    plt.legend()  # Graph ke liye legend (label) dikhate hain
    plt.show()  # Graph display karte hain

def plot_predicted_data(predicted_data):
    plt.figure(figsize=(10, 5))  # Graph ka size set karte hain (width=10, height=5)
    plt.plot(predicted_data['date'], predicted_data['predicted_temperature'], color='green', linestyle='--', marker='x', label='Predicted Avg Temperature')  
    # Predicted average temperature ko plot karte hain (future dates x-axis par aur predicted temperature y-axis par)
    plt.title('Predicted Temperature (Next 7 Days)')  # Graph ka title set karte hain
    plt.xlabel('Date')  # X-axis ka label 'Date' set karte hain
    plt.ylabel('Temperature (°C)')  # Y-axis ka label 'Temperature (°C)' set karte hain
    plt.grid(True)  # Graph me grid lines dikhate hain
    plt.legend()  # Graph ke liye legend (label) dikhate hain
    plt.tight_layout()  # Graph ke layout ko adjust karte hain taake sab kuch properly dikhe
    plt.xticks(predicted_data['date'], predicted_data['date'].dt.strftime('%d-%b-%Y'), rotation=45)  
    # X-axis par dates dikhate hain formatted style me (DD-MMM-YYYY) aur unko thoda rotate karte hain taake overlap na ho
    plt.show()  # Graph display karte hain


def show_graph():
    if predicted_data is not None:
        plot_predicted_data(predicted_data)  # Agar predicted data hai to graph dikhaye
    else:
        tk.messagebox.showerror("Error", "No predicted data available to show.")  # Agar data nahi hai to error dikhaye

def show_historical_graph():
    if historical_data is not None:
        plot_historical_data(historical_data)  # Agar historical data hai to graph dikhaye
    else:
        tk.messagebox.showerror("Error", "No historical data available to show.")  # Agar data nahi hai to error dikhaye

# Function jo background image ko dynamically update kare
def update_background(image_path, label):
    bg_image = Image.open(image_path)  # Image ko open karte hain #winfo.width returns label horizontal size #winfo.height returns label vertical size
    bg_image = bg_image.resize((label.winfo_width(), label.winfo_height()), Image.Resampling.LANCZOS)  # Image ka size label ke size ke barabar set karte hain
    bg_photo = ImageTk.PhotoImage(bg_image)  # Image ko PhotoImage format mein convert karte hain
    label.configure(image=bg_photo)  # Label pe image set karte hain
    label.image = bg_photo  # Image ka reference label ke saath store karte hain


# Function jo application ko exit kare
def exit_application():
    root.withdraw()  # Main window ko hide karo
    exit_window = tk.Toplevel()  # Naya window banaye
    exit_window.title("Exit Confirmation")
    exit_window.geometry("400x200")
    exit_window.configure(bg='#00AFFF')
    #pack() tkinter ka layout manager hai jo widgets ko window mein arrange karta hai. Yeh widgets ko vertically (upar se neeche) ya horizontally (left se right) place karta hai, aur unka size automatically adjust kar leta hai.
    tk.Label(exit_window, text="Thank you for using the Weather Prediction AI!\nWe hope our system helped you with accurate weather forecasts.\nHave a great day!", 
             font=("Times New Roman", 14), bg='#00AFFF', fg='white').pack(pady=20)  # Exit message dikhaye

# Function jo linear regression ka kaam dikhaye
def show_linear_regression_working():
    with open("../linearreg.txt", "r") as file:
        content = file.read()  # File se content read kare

    # Naya window banaye content dikhane ke liye
    working_window = tk.Toplevel(root)
    working_window.title("Working of Linear Regression")
    working_window.geometry("900x800")
    working_window.configure(bg='#00AFFF')

    # Canvas aur scrollbar banaye
    canvas = tk.Canvas(working_window, bg='#00AFFF')
    scrollbar = tk.Scrollbar(working_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='#00AFFF')

    #Canvas ko configure kare
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Scrollbar ko right side par place kare
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Scrollable frame mein content add kare
    lines = content.splitlines()
    
    #enumerate() ka use yahan isliye ho raha hai taake har line ke saath uska index bhi mile.
    # Har line ke liye labels banaye formatting ke sath
    for index, line in enumerate(lines):
        if "Linear Regression for Weather Prediction AI" in line:
            tk.Label(scrollable_frame, text=line, font=("Times New Roman", 14, "bold"), bg='white').grid(row=index, column=0, sticky='w', padx=10, pady=5)
        elif "Introduction to Linear Regression in Weather Prediction" in line:
            tk.Label(scrollable_frame, text=line, font=("Times New Roman", 14, "bold"), bg='white').grid(row=index, column=0, sticky='w', padx=10, pady=5)
        elif "The Linear Regression Equation" in line:
            tk.Label(scrollable_frame, text=line, font=("Times New Roman", 14, "bold"), bg='white').grid(row=index, column=0, sticky='w', padx=10, pady=5)
        else:
            tk.Label(scrollable_frame, text=line, font=("Times New Roman", 12), bg='white').grid(row=index, column=0, sticky='w', padx=10, pady=2)

    # Scrollbar ko canvas ke sath configure kare
    canvas.configure(yscrollcommand=scrollbar.set)

# GUI setup
root = tk.Tk()
root.title("Weather Forecast Application")
root.geometry("800x600")
root.configure(bg='#00AFFF')

# Background label
bg_label = tk.Label(root)
bg_label.place(relwidth=1, relheight=1)

update_background("../weatherbg.PNG", bg_label)  # Background image update kare

# Title label
tk.Label(root, text="Weather Prediction Ai", font=("Times New Roman ", 24, "bold"), bg='#00AFFF', fg='white').pack(pady=20)

# "Enter City" label
tk.Label(root, text="Enter City:", bg='#00AFFF', fg='white', font=("Times New Roman", 13)).pack(pady=5)

# City input ke liye entry widget
city_entry = tk.Entry(root, font=("Times New Roman", 14), width=20)
city_entry.pack(pady=5)

# Date Entry
tk.Label(root, text="Enter Date from only next 7 days(date-month-year) [Optional]:", bg='#00AFFF', fg='white', font=("Times New Roman", 13)).pack(pady=5)
date_entry = tk.Entry(root, font=("Times New Roman", 14), width=20)
date_entry.pack(pady=5)

# Search Button
tk.Button(root, text="Temperature Prediction", bg="white", fg="black", command=show_predictions).pack(pady=20)  # Temperature prediction button

# Show Predicted Graph Button
tk.Button(root, text="Show Predicted Graph", bg="white", fg="black", command=show_graph).pack(pady=10)  # Predicted graph button

# Show Historical Weather Graph Button
tk.Button(root, text="Show Historical Weather Graph", bg="white", fg="black", command=show_historical_graph).pack(pady=10)  # Historical weather graph button

# Working of Linear Regression Button
tk.Button(root, text="Working of Linear Regression", bg="white", fg="black", command=show_linear_regression_working).pack(pady=10)  # Linear regression working button

# Exit Button
tk.Button(root, text="Exit", bg="white", fg="black", command=exit_application).pack(pady=10)  # Exit button
# configure se hum widget ki bg color,font size ya koi b property modify krsakte
root.bind("<Configure>", lambda e: update_background("weatherbg.PNG", bg_label))  # Resize par background update kare

root.mainloop()  # Main loop ko start kare
