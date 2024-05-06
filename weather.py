#!/usr/bin/env python
# coding: utf-8

# ## Disclaimer: HTML Content Changes
# 
# Please note that HTML content structure and tags on websites may change over time, leading to potential discrepancies in the code's functionality. If the script fails to produce the expected results, it could indicate alterations in the HTML code of the webpage being scraped. In such cases, the code may need to be updated to accommodate the changes in the webpage's structure or tags.

# ## Installations required for the project

import subprocess

# ## Imports required for the project

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# # URL 1 https://forecast.weather.gov

# # Weather Forecast Data Analysis
# 
# ## Introduction
# This project involves scraping weather forecast data from a website, processing it, and visualizing various aspects of the forecast using Python. The code retrieves the forecast data, extracts relevant information such as period names, short descriptions, and temperatures, performs data analysis, and finally creates visualizations to provide insights into the weather forecast.
# 
# ## Code Explanation
# The code consists of the following main sections:
# 
# ### Imports
# - Necessary libraries are imported, including `requests`, `BeautifulSoup`, `pandas`, `re`, `datetime`, `seaborn`, and `matplotlib.pyplot`. The `warnings` module is also imported to suppress warnings for cleaner output.
# 
# ### Data Retrieval and Processing
# - The code sends a GET request to the URL containing the weather forecast and parses the HTML content using BeautifulSoup.
# - Forecast items such as period names, short descriptions, and temperatures are extracted from the parsed HTML content.
# - The extracted data is stored in a pandas DataFrame and saved to a CSV file.
# 
# ### Data Analysis
# - The CSV file is read back into a pandas DataFrame for further analysis.
# - Regular expressions are used to extract numeric temperature values and temperature types (high/low) from the 'Temperature' column.
# - Descriptive statistics of temperature values such as mean, mode, and median are calculated.
# 
# ### Data Visualization
# - Various visualizations are created using matplotlib.pyplot:
#   - Bar chart showing temperature highs for each period.
#   - Bar chart showing temperature lows for each period.
#   - Line plot showing the temperature trend over the forecast period.
#   - Pie chart showing the distribution of weather conditions (short descriptions) over the forecast period.
#   - Histogram showing the distribution of temperature lows.
# 
# Each visualization provides insights into different aspects of the weather forecast, aiding in better understanding and interpretation of the data.
# 
# ## Conclusion
# This project demonstrates how to retrieve, process, analyze, and visualize weather forecast data using Python. By analyzing and visualizing the data, we gain valuable insights into the forecasted weather conditions, helping us make informed decisions.

url = "https://forecast.weather.gov/MapClick.php?lat=41.884250000000065&lon=-87.63244999999995#.XtpdeOfhXIX"
r = requests.get(url)

soup = BeautifulSoup(r.content,"html.parser")
week = soup.find(id="seven-day-forecast-body")
items = soup.find_all("div",class_ = "tombstone-container")
period_name = [item.find(class_="period-name").get_text() for item in items]
short_desc = [item.find(class_="short-desc").get_text() for item in items]
temp = [item.find(class_="temp").get_text() for item in items]

df = pd.DataFrame({"Period" : period_name,"Short Description" : short_desc,"Temperature" : temp})
df.to_csv("18IT023_WeatherData.csv")

# Read the CSV file into a DataFrame
data = pd.read_csv('18IT023_WeatherData.csv', encoding='utf-8')

# Extract temperature values and types (high/low) using regular expressions
data['Temperature'] = data['Temperature'].astype(str)
data['Temp Value'] = data['Temperature'].str.extract('(\d+)').astype(int)
data['Temp Type'] = data['Temperature'].str.extract('([a-zA-Z]+)')

# Display the modified dataframe
print(data[['Period', 'Temperature', 'Temp Value', 'Temp Type']].head())

# Display descriptive statistics of temperature values
print(data['Temp Value'].describe())
print("Mean:", data['Temp Value'].mean())
print("Mode:", data['Temp Value'].mode())
print("Median:", data['Temp Value'].median())

# Filter the DataFrame to obtain only the rows corresponding to low temperatures
low_temp_data = data[data['Temp Type'] == 'Low']

# 1. Bar chart showing the temperature highs for each period
plt.figure(figsize=(10, 6))
plt.bar(data['Period'], data['Temp Value'], color='red')
plt.xlabel('Period')
plt.ylabel('Temperature High (°F)')
plt.title('Temperature Highs for Each Period')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('temperature_highs.png')  # Save the plot as an image file
plt.show()

# 2. Bar chart showing the temperature lows for each period
plt.figure(figsize=(10, 6))
plt.bar(low_temp_data['Period'], low_temp_data['Temp Value'], color='lightgreen')
plt.xlabel('Period')
plt.ylabel('Temperature Low (°F)')
plt.title('Temperature Lows for Each Period')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('temperature_lows.png')  # Save the plot as an image file
plt.show()

# 3. Line plot showing the temperature trend over the forecast period
plt.figure(figsize=(10, 6))
plt.plot(data['Period'], data['Temp Value'], marker='o', color='orange')
plt.xlabel('Period')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Trend Over the Forecast Period')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('temperature_trend.png')  # Save the plot as an image file
plt.show()

# 4. Pie chart showing the distribution of weather conditions (short descriptions) over the forecast period
plt.figure(figsize=(8, 8))
plt.pie(data['Short Description'].value_counts(), labels=data['Short Description'].unique(), autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Weather Conditions')
plt.axis('equal')
plt.tight_layout()
plt.savefig('weather_conditions.png')  # Save the plot as an image file
plt.show()

# 5. Histogram showing the distribution of temperature lows
plt.figure(figsize=(8, 6))
plt.hist(low_temp_data['Temp Value'], bins=10, color='lightgreen', edgecolor='black')
plt.xlabel('Temperature Low (°F)')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature Lows')
plt.tight_layout()
plt.savefig('temperature_lows_histogram.png')  # Save the plot as an image file
plt.show()


print("-----------------------------------------------------------end of url--------------------------------------------------------------------------------------")
#!/usr/bin/env python
# coding: utf-8

# ## Disclaimer: HTML Content Changes
# 
# Please note that HTML content structure and tags on websites may change over time, leading to potential discrepancies in the code's functionality. If the script fails to produce the expected results, it could indicate alterations in the HTML code of the webpage being scraped. In such cases, the code may need to be updated to accommodate the changes in the webpage's structure or tags.

# ## Installations required for the project

# Define the URL 2
url = "http://www.estesparkweather.net/archive_reports.php?date="

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Set user-agent headers to avoid blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

url = "http://www.estesparkweather.net/archive_reports.php?date="

# Define function to scrape data
def scrape_weather_data(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find_all('table')
        raw_data = [row.text.splitlines() for row in table]
        raw_data = raw_data[:-9]
        for i in range(len(raw_data)):
            raw_data[i] = raw_data[i][2:len(raw_data[i]):3]
        
        # Scrape data and return
        df_list = []
        index = []
        for i in range(len(raw_data)):
            c = ['.'.join(re.findall("\d+", str(raw_data[i][j].split()[:5]))) for j in range(len(raw_data[i]))]
            df_list.append(c)
            index.append(dates[k] + c[0])
        
        # Filter and process data
        f_index = [index[i] for i in range(len(index)) if len(index[i]) > 6]
        data = [df_list[i][1:] for i in range(len(df_list)) if len(df_list[i][1:]) == 19]
        final_index = [datetime.strptime(str(f_index[i]), '%Y%m%d').strftime('%Y-%m-%d') for i in range(len(f_index))]
        
        return final_index, data
    else:
        print('Failed to fetch weather data for:', url)
        return None, None

# Define date range for scraping (01-01-2024 to 02-01-2024)
Dates_r = pd.date_range(start='2024-01-01', end='2024-02-01', freq='M')
dates = [str(i)[:4] + str(i)[5:7] for i in Dates_r]

# Define lists to store scraped data
final_index_list = []
data_list = []

# Iterate over dates and scrape data
for k in range(len(dates)):
    url = "http://www.estesparkweather.net/archive_reports.php?date="
    url += dates[k]
    final_index, data = scrape_weather_data(url)
    if final_index and data:
        final_index_list.extend(final_index)
        data_list.extend(data)

# Create DataFrame
df = pd.DataFrame(data_list, index=final_index_list, columns=[f'Data_{i}' for i in range(1, 20)])

# Save DataFrame to CSV file
df.to_csv('weather_data_url2.csv')

# Dictionary to map current column names to new names
column_mapping = {
    "Data_1": "Average temperature",
    "Data_2": "Average humidity",
    "Data_3": "Average dewpoint",
    "Data_4": "Average barometer",
    "Data_5": "Average windspeed",
    "Data_6": "Average gustspeed",
    "Data_7": "Average direction",
    "Data_8": "Rainfall for month",
    "Data_9": "Rainfall for year",
    "Data_10": "Maximum rain per minute",
    "Data_11": "Maximum temperature",
    "Data_12": "Minimum temperature",
    "Data_13": "Maximum humidity",
    "Data_14": "Minimum humidity",
    "Data_15": "Maximum pressure",
    "Data_16": "Minimum pressure",
    "Data_17": "Maximum windspeed",
    "Data_18": "Maximum gust speed",
    "Data_19": "Maximum heat index"
}

# Rename columns
df = df.rename(columns=column_mapping)

# Function to clean and convert string values to numeric
def clean_and_convert(value):
    # Remove non-numeric characters
    cleaned_value = ''.join(e for e in str(value) if e.isdigit() or e == '.')
    # Convert to numeric
    try:
        return float(cleaned_value)
    except ValueError:
        return None

# Apply the function to all columns
df = df.applymap(clean_and_convert)

# Print summary statistics
print(df.describe())

# Print mean for numeric columns
numeric_columns = ['Average temperature', 'Average humidity', 'Average dewpoint', 'Average barometer',
                   'Average windspeed', 'Average gustspeed', 'Maximum temperature', 'Minimum temperature',
                   'Maximum humidity', 'Minimum humidity', 'Maximum pressure', 'Minimum pressure',
                   'Maximum windspeed', 'Maximum gust speed', 'Maximum heat index']

print(df[numeric_columns].mean())
print(df.median())
print(df.min())
print(df.max())

# Plotting histograms
# Plotting histogram for 'Average temperature'
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
df['Average temperature'].hist(ax=ax, bins=20, alpha=0.7, color='blue')
ax.set_title('Histogram of Average Temperature')
ax.set_xlabel('Temperature')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('average_temperature_histogram.png')  # Save the histogram as a PNG file
plt.show()

# Plotting histogram for 'Average humidity'
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
df['Average humidity'].hist(ax=ax, bins=20, alpha=0.7, color='blue')
ax.set_title('Histogram of Average Humidity')
ax.set_xlabel('Humidity')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('average_humidity_histogram.png')  # Save the histogram as a PNG file
plt.show()

# Plotting histogram for 'Average windspeed'
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
df['Average windspeed'].hist(ax=ax, bins=20, alpha=0.7, color='blue')
ax.set_title('Histogram of Average Windspeed')
ax.set_xlabel('Windspeed')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('average_windspeed_histogram.png')  # Save the histogram as a PNG file
plt.show()

# Convert 'Average temperature' to numeric
df['Average temperature'] = pd.to_numeric(df['Average temperature'], errors='coerce')

# Convert 'Average humidity' to numeric
df['Average humidity'] = pd.to_numeric(df['Average humidity'], errors='coerce')

# Define the variables to plot
vars_to_plot = ['Average temperature', 'Average humidity']

# Check if the specified variables are in the DataFrame
if all(var in df.columns for var in vars_to_plot):
    # Set up the figure with white background
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), facecolor='white')

    # Plotting boxplot for Average temperature
    sns.boxplot(y=df['Average temperature'], ax=axes[0], color='skyblue')
    axes[0].set_title('Boxplot of Average Temperature')
    axes[0].set_ylabel('Temperature')
    plt.savefig('boxplot_average_temperature.png')  # Save the boxplot as a PNG file

    # Plotting violin plot for Average humidity
    sns.violinplot(y=df['Average humidity'], ax=axes[1], color='lightgreen')
    axes[1].set_title('Violin Plot of Average Humidity')
    axes[1].set_ylabel('Humidity')
    plt.savefig('violinplot_average_humidity.png')  # Save the violin plot as a PNG file

    plt.tight_layout()
    plt.show()
else:
    print("One or more specified variables are not present in the DataFrame.")


print("-----------------------------------------------------------end of url--------------------------------------------------------------------------------------")
# # URL 3: Extracting hourly data from the website https://www.bbc.com/weather 

# # Extracting Hourly Weather Data from https://www.bbc.com/weather
# 
# To fetch hourly weather data from the website https://www.bbc.com/weather, we utilize Python with the `requests` library for HTTP requests and `BeautifulSoup` from `bs4` for HTML parsing. Additionally, `pandas` is employed for data manipulation and analysis.
# 
# ## Code Explanation
# 
# ### Data Retrieval
# - The function `scrape_bbc_weather(location)` constructs the URL for the specified location and sends a GET request to it.
# - It then parses the HTML content of the page to extract weather information for each hour.
# 
# ### Data Processing
# - The extracted weather data, including time, temperature, and description, are organized into a pandas DataFrame.
# 
# ### Data Analysis
# - Basic statistical analysis such as mean, median, minimum, maximum, and standard deviation of temperature is performed.
# 
# ### Data Visualization
# - Several plots are generated using `seaborn` to visualize the data:
#   - Line plot illustrating temperature variation over time.
#   - Histogram displaying the distribution of temperatures.
#   - Box plot visualizing temperature distribution.
#   - Violin plot comparing temperature distributions across different times of the day.
#   - Bar plot showing the frequency of different weather descriptions.
# 
# ## Conclusion
# 
# This code demonstrates how to efficiently retrieve hourly weather data from the BBC Weather website, process it, perform basic analysis, and create insightful visualizations using Python. The resulting insights can be invaluable for understanding temperature trends, weather patterns, and making informed decisions in various domains.
# 

# In[29]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_bbc_weather(location):
    # Construct the URL for the specified location
    url = f"https://www.bbc.com/weather/{location}"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract weather information from the page
        weather_data = []
        # Find the container with class "wr-time-slot-primary" which contains the weather information
        weather_containers = soup.find_all(class_="wr-time-slot-primary")
        for container in weather_containers:
            # Extract relevant weather details such as time, temperature, and description
            time = container.find(class_="wr-time-slot-primary__time").get_text(strip=True)
            temperature = container.find(class_="wr-value--temperature--c").get_text(strip=True)
            description = container.find(class_="wr-time-slot-primary__weather-type-description").get_text(strip=True)
            
            # Append the extracted data to the weather_data list
            weather_data.append({"Time": time, "Temperature (°C)": temperature, "Description": description})
        
        # Create a DataFrame from the weather_data list
        df = pd.DataFrame(weather_data)
        
        return df
    else:
        print(f"Failed to retrieve data for {location}. Status code: {response.status_code}")

# Example usage:
location = "5128581"  # New York (You can find the location ID in the URL of the weather page)
weather_df = scrape_bbc_weather(location)
print(weather_df)


# In[30]:


weather_df.info()


# In[31]:


weather_df.dtypes


# In[32]:


# Remove the degree symbol and convert to numeric, handling errors by coercing to NaN
weather_df['Temperature (°C)'] = pd.to_numeric(weather_df['Temperature (°C)'].str.replace('°', ''), errors='coerce')


# In[33]:


# Perform basic statistical analysis
mean_temperature = weather_df['Temperature (°C)'].mean()
median_temperature = weather_df['Temperature (°C)'].median()
min_temperature = weather_df['Temperature (°C)'].min()
max_temperature = weather_df['Temperature (°C)'].max()
std_dev_temperature = weather_df['Temperature (°C)'].std()

# Print the results
print(f"Mean Temperature: {mean_temperature:.2f}°C")
print(f"Median Temperature: {median_temperature:.2f}°C")
print(f"Minimum Temperature: {min_temperature:.2f}°C")
print(f"Maximum Temperature: {max_temperature:.2f}°C")
print(f"Standard Deviation of Temperature: {std_dev_temperature:.2f}°C")


# In[101]:


# Set the style of the plots
sns.set_style("whitegrid")

# Create a directory to save the plots if it doesn't exist
if not os.path.exists("weather_plots"):
    os.makedirs("weather_plots")

# Plot 1: Line plot showing the variation of temperature over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='Temperature (°C)', data=weather_df, marker='o')
plt.title('Temperature Variation Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plot_filename = "weather_plots/temperature_variation_over_time.png"
plt.savefig(plot_filename)
plt.show()

# Plot 2: Histogram of temperature distribution
plt.figure(figsize=(8, 6))
sns.histplot(weather_df['Temperature (°C)'], bins=10, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.tight_layout()
plot_filename = "weather_plots/temperature_distribution_histogram.png"
plt.savefig(plot_filename)
plt.show()

# Plot 3: Box plot to visualize the distribution of temperatures
plt.figure(figsize=(8, 6))
sns.boxplot(y='Temperature (°C)', data=weather_df)
plt.title('Temperature Distribution')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plot_filename = "weather_plots/temperature_distribution_boxplot.png"
plt.savefig(plot_filename)
plt.show()

# Plot 4: Violin plot to compare the temperature distributions across different times of the day
plt.figure(figsize=(10, 6))
sns.violinplot(x='Time', y='Temperature (°C)', data=weather_df, inner='quartile')
plt.title('Temperature Distribution Across Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plot_filename = "weather_plots/temperature_distribution_violinplot.png"
plt.savefig(plot_filename)
plt.show()

# Plot 5: Bar plot showing the frequency of different weather descriptions
plt.figure(figsize=(10, 6))
sns.countplot(y='Description', data=weather_df, order=weather_df['Description'].value_counts().index)
plt.title('Frequency of Weather Descriptions')
plt.xlabel('Frequency')
plt.ylabel('Weather Description')
plt.tight_layout()
plot_filename = "weather_plots/weather_description_frequency.png"
plt.savefig(plot_filename)
plt.show()

print("Plots saved successfully in the 'weather_plots' directory!")


# In[102]:


# Save the DataFrame as a CSV file
weather_df.to_csv("weather_data_url4.csv", index=False)

print("DataFrame saved successfully as 'weather_data.csv'!")

print("-----------------------------------------------------------end of url--------------------------------------------------------------------------------------")
# # URL 4 Here there are changes in the listing every couple of days

# # Retrieving Weather Data from Google
# 
# This Python script aims to fetch weather data from Google for various cities in the United States. It utilizes `BeautifulSoup` for HTML parsing, `requests` for making HTTP requests, `pandas` for data manipulation, and `matplotlib` for data visualization.
# 
# ## Code Overview
# 
# ### Weather Data Extraction
# - The function `find_weather(city_name)` queries Google for weather data of a specified city.
# - It parses the HTML response to extract location, temperature, time, and weather description.
# 
# ### Fetching Data for Multiple States
# - The function `get_states_weather(states)` iterates through a list of major states and retrieves weather data for each state.
# 
# ### Data Comparison
# - The function `compare_datasets(previous_data, current_data)` compares the current day's weather dataset with the previous day's dataset.
# - It identifies new, removed, and updated rows between the two datasets.
# 
# ### Data Processing
# - After retrieving weather data, it updates the "Weather Description" column to keep only the description part.
# - It drops the "Location" column and renames the "Time" column to "Location".
# - Additionally, it filters out rows based on a regex pattern for the expected format of the "Location" column.
# 
# ### Data Visualization
# - Several plots are generated to visualize the weather data:
#   - Temperature distribution histogram.
#   - Bar plot showing the frequency of different weather descriptions.
#   - Scatter plot illustrating temperature variation across different locations.
#   - Bar plots for the top 5 locations with the highest and lowest temperatures.
# 
# ### Data Storage
# - The current day's weather data is saved to a CSV file named "weather_data_previous.csv" for future comparison.
# 
# ## Conclusion
# 
# This script demonstrates a practical approach to fetch and analyze weather data from Google for multiple locations. By comparing daily datasets, processing, and visualizing the information, it provides insights into temperature variations and weather conditions across different regions, aiding in decision-making and understanding weather trends.
# 

# In[34]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Set user-agent headers to avoid blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def find_weather(city_name):
    try:
        # Replace spaces in the city name with '+'
        city_name = city_name.replace(" ", "+")

        # Make a query to Google for weather data
        res = requests.get(f"https://www.google.com/search?q=weather+{city_name}", headers=headers)

        # Parse the HTML response
        soup = BeautifulSoup(res.text, 'html.parser')

        # Extract relevant weather information
        location = soup.select('.BNeawe.iBp4i.AP7Wnd')[0].getText()
        temperature = soup.select('.BNeawe.iBp4i.AP7Wnd')[1].getText()
        time = soup.select('.BNeawe.tAd8D.AP7Wnd')[0].getText()
        info = soup.select('.BNeawe.tAd8D.AP7Wnd')[1].getText()

        return {
            "Location": location,
            "Temperature": temperature,
            "Time": time,
            "Weather Description": info
        }
    except Exception as e:
        print("Error:", e)

def get_states_weather(states):
    weather_data = []
    for state in states:
        weather = find_weather(state)
        if weather:
            weather_data.append(weather)
    return weather_data

# List of major states
states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
          "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
          "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
          "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
          "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
          "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]

def compare_datasets(previous_data, current_data):
    """
    Compare the previous day's dataset with the current day's dataset.
    
    Args:
        previous_data (pd.DataFrame): DataFrame containing the previous day's data.
        current_data (pd.DataFrame): DataFrame containing the current day's data.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'new_rows': List of new rows in the current data.
            - 'removed_rows': List of rows removed from the previous data.
            - 'updated_rows': List of rows with changes in the current data.
    """
    # Create a set of indices for each dataset
    previous_indices = set(previous_data.index)
    current_indices = set(current_data.index)
    
    # Find new rows
    new_rows = [current_data.loc[idx] for idx in current_indices - previous_indices]
    
    # Find removed rows
    removed_rows = [previous_data.loc[idx] for idx in previous_indices - current_indices]
    
    # Find updated rows
    common_indices = previous_indices.intersection(current_indices)
    updated_rows = []
    for idx in common_indices:
        previous_row = previous_data.loc[idx]
        current_row = current_data.loc[idx]
        if not previous_row.equals(current_row):
            updated_rows.append((previous_row, current_row))
    
    return {
        'new_rows': new_rows,
        'removed_rows': removed_rows,
        'updated_rows': updated_rows
    }

# Get current day's weather data
weather_data = get_states_weather(states)
df = pd.DataFrame(weather_data)

# Load previous day's data (if available)
try:
    previous_data = pd.read_csv("weather_data_previous.csv")
    # Compare datasets
    changes = compare_datasets(previous_data, df)

    # Print changes
    print("New rows:")
    print(changes['new_rows'])
    print("\nRemoved rows:")
    print(changes['removed_rows'])
    print("\nUpdated rows:")
    for previous_row, current_row in changes['updated_rows']:
        print(f"Previous: {previous_row}")
        print(f"Current: {current_row}")
        print()
except FileNotFoundError:
    print("Previous data not found. Skipping comparison.")

# Update the "Weather Condition" column to keep only the second part after splitting by '\n'
df['Weather Description'] = df['Weather Description'].str.split('\n').str[1]

# Drop the "Location" column
df.drop(columns=['Location'], inplace=True)

# Rename the "Time" column to "Location"
df.rename(columns={'Time': 'Location'}, inplace=True)

# Define regex pattern for the expected format of the "Location" column
location_pattern = r'^[A-Z][a-z]+,\s[A-Z]{2}$'

# Filter rows based on regex pattern
df_filtered = df[df['Location'].str.match(location_pattern)]

# Extract temperature values and convert to float
df['Temperature'] = df['Temperature'].str.extract(r'(\d+\.?\d*)').astype(float)


# In[36]:


mean_temperature = df['Temperature'].mean()
median_temperature = df['Temperature'].median()
min_temperature = df['Temperature'].min()
max_temperature = df['Temperature'].max()
std_dev_temperature = df['Temperature'].std()

# Print the results
print(f"Mean Temperature: {mean_temperature:.2f}°C")
print(f"Median Temperature: {median_temperature:.2f}°C")
print(f"Minimum Temperature: {min_temperature:.2f}°C")
print(f"Maximum Temperature: {max_temperature:.2f}°C")
print(f"Standard Deviation of Temperature: {std_dev_temperature:.2f}°C")


# In[37]:


# Plot 1: Temperature distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Temperature'], bins=10, color='skyblue', edgecolor='black')
plt.title('Temperature Distribution')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('temperature_distribution.png')  # Save the figure
plt.show()
plt.close()

# Plot 2: Weather description frequency
plt.figure(figsize=(10, 6))
df['Weather Description'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Weather Description Frequency')
plt.xlabel('Weather Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.savefig('weather_description_frequency.png')  # Save the figure
plt.show()
plt.close()

# Plot 3: Temperature vs Location
plt.figure(figsize=(14, 8))
plt.scatter(df['Location'], df['Temperature'], color='orange')
plt.title('Temperature vs Location')
plt.xlabel('Location')
plt.ylabel('Temperature (°F)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.savefig('temperature_vs_location.png')  # Save the figure
plt.show()
plt.close()

# Plot 4: Top 5 Locations with highest temperature
top_locations = df.nlargest(5, 'Temperature')
plt.figure(figsize=(10, 6))
plt.bar(top_locations['Location'], top_locations['Temperature'], color='salmon')
plt.title('Top 5 Locations with Highest Temperature')
plt.xlabel('Location')
plt.ylabel('Temperature')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.savefig('top_5_locations_highest_temperature.png') 
plt.show()
plt.close()

# Plot 5: Top 5 Locations with lowest temperature
bottom_locations = df.nsmallest(5, 'Temperature')
plt.figure(figsize=(10, 6))
plt.bar(bottom_locations['Location'], bottom_locations['Temperature'], color='lightblue')
plt.title('Top 5 Locations with Lowest Temperature')
plt.xlabel('Location')
plt.ylabel('Temperature')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.savefig('top_5_locations_lowest_temperature.png')  # Save the figure
plt.show()
plt.close()

# Save current day's data to CSV
df.to_csv("weather_data_previous.csv", index=False)
print("Current weather data saved to weather_data_previous.csv")

print("-----------------------------------------------------------end of url--------------------------------------------------------------------------------------")
# # URL 5: www.weather-atlas.com 

# # Web Scraping Weather Data from Weather-Atlas.com
# 
# This project aims to scrape weather data for Paris, France, from the website [weather-atlas.com](https://www.weather-atlas.com/en/france/paris-long-term-weather-forecast) and perform basic data analysis and visualization on the extracted information.
# 
# ## Code Overview
# 
# 1. **Web Scraping**: The code uses the `requests` and `BeautifulSoup` libraries to fetch the HTML content of the website and extract relevant weather data from the page.
# 
# 2. **Data Cleaning**: Several helper functions (`clean_data`, `clean_wind`, and `clean_other`) are defined to clean and transform the extracted data into a suitable format for analysis.
# 
# 3. **Data Storage**: The cleaned weather data is stored in a Pandas DataFrame named `weather_df`.
# 
# 4. **Data Preprocessing**: The 'Weather Condition' column is dropped from the DataFrame since it contains no useful information.
# 
# 5. **Statistical Analysis**: A function `calculate_stats` is defined to compute various statistical measures (minimum, maximum, mean, median, and standard deviation) for the numerical columns in the DataFrame.
# 
# 6. **Data Visualization**: The code uses the `matplotlib` and `seaborn` libraries to generate the following plots:
#    - **Normal Distribution Plots**: Histogram plots with kernel density estimation (KDE) for the numerical columns.
#    - **Skewness Plots**: Box plots to visualize the skewness of the data for each numerical column.
#    - **Scatter Plots**: Scatter plots for every pair of numerical columns.
#    - **Pair Plot**: A matrix of scatter plots and KDE plots for all pairs of numerical columns.
# 
# 7. **File Output**: The original `weather_df` DataFrame is saved as a CSV file named 'weather-atlas.csv', and the generated plots are saved as PNG files in the same directory as the Python script.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def clean_data(text):
    # Remove non-digit characters and strip whitespace
    cleaned_text = re.sub(r'\D', '', text).strip()
    return int(cleaned_text) if cleaned_text else None

def clean_wind(text):
    # Extract wind speed and direction using regex
    match = re.match(r'Wind:(\d+)km/h(\w+)', text)
    if match:
        return int(match.group(1)), match.group(2)
    else:
        return None, None

def clean_other(text):
    # Extract numeric value using regex
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else None

# Get the HTML content from the website
file = requests.get("https://www.weather-atlas.com/en/france/paris-long-term-weather-forecast")
soup = BeautifulSoup(file.content, "html.parser")

# Find all the rows containing weather data
weather_rows = soup.find_all("div", class_="row px-3 py-2 border-bottom")

# Initialize lists to store data
data = {
    'Date': [],
    'Weather Condition': [],
    'Max Temperature': [],
    'Min Temperature': [],
    'Wind Speed (km/h)': [],
    'Wind Direction': [],
    'Humidity (%)': [],
    'Precipitation Probability (%)': [],
    'Precipitation Amount (%)': []
}

# Iterate through each row to extract weather information
for row in weather_rows:
    # Extract date
    date_element = row.find("span", class_="d-block fw-bold")
    date = date_element.get_text(strip=True) if date_element else "Date not available"
    data['Date'].append(date)
    
    # Extract weather condition
    weather_condition_element = row.find("span", class_="fs-4 fw-bold")
    weather_condition = weather_condition_element.get_text(strip=True) if weather_condition_element else "Weather condition not available"
    data['Weather Condition'].append(weather_condition)
    
    # Extract temperature
    temperature_elements = row.find_all("li")
    max_temp = clean_data(temperature_elements[0].get_text(strip=True)) if temperature_elements else None
    min_temp = clean_data(temperature_elements[1].get_text(strip=True)) if len(temperature_elements) > 1 else None
    data['Max Temperature'].append(max_temp)
    data['Min Temperature'].append(min_temp)
    
    # Extract wind information
    wind_text = temperature_elements[2].get_text(strip=True) if len(temperature_elements) > 2 else None
    wind_speed, wind_direction = clean_wind(wind_text) if wind_text else (None, None)
    data['Wind Speed (km/h)'].append(wind_speed)
    data['Wind Direction'].append(wind_direction)
    
    # Extract humidity, precipitation probability, and precipitation amount
    humidity_text = temperature_elements[3].get_text(strip=True) if len(temperature_elements) > 3 else None
    precip_prob_text = temperature_elements[4].get_text(strip=True) if len(temperature_elements) > 4 else None
    precip_amount_text = temperature_elements[5].get_text(strip=True) if len(temperature_elements) > 5 else None
    
    data['Humidity (%)'].append(clean_other(humidity_text))
    data['Precipitation Probability (%)'].append(clean_other(precip_prob_text))
    data['Precipitation Amount (%)'].append(clean_other(precip_amount_text))

# Create a DataFrame
weather_df = pd.DataFrame(data)

# Display the DataFrame
print(weather_df)


# In[4]:


weather_df


# In[5]:


weather_df = weather_df.drop(columns=['Weather Condition'])


# In[6]:


weather_df


# In[18]:


def calculate_stats(weather_df):
    numeric_columns = ['Max Temperature', 'Min Temperature', 'Wind Speed (km/h)', 'Humidity (%)', 'Precipitation Probability (%)', 'Precipitation Amount (%)']
    stats_dict = {}

    for column in numeric_columns:
        stats_dict[column] = {
            'Min': weather_df[column].min(),
            'Max': weather_df[column].max(),
            'Mean': weather_df[column].mean(),
            'Median': weather_df[column].median(),
            'Std': weather_df[column].std()
        }
    
    return stats_dict

# Calculate statistics
stats = calculate_stats(weather_df)
for column, values in stats.items():
    print(f"{column}:")
    for stat, value in values.items():
        print(f"  {stat}: {value}")
    print()


# In[7]:


weather_df.to_csv('weather-atlas.csv', index=False)


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Normal Distribution Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Max Temperature', 'Min Temperature', 'Wind Speed (km/h)', 'Humidity (%)', 'Precipitation Probability (%)']):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=weather_df, x=col, kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig('normal_distribution_plots.png', dpi=300)

# Skewness Plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(['Max Temperature', 'Min Temperature', 'Wind Speed (km/h)', 'Humidity (%)', 'Precipitation Probability (%)']):
    plt.subplot(2, 3, i+1)
    sns.boxplot(data=weather_df, x=col)
    plt.title(f'{col} Skewness Plot')
plt.tight_layout()
plt.savefig('skewness_plots.png', dpi=300)

# Scatter Plots
plt.figure(figsize=(12, 8))
cols = ['Max Temperature', 'Min Temperature', 'Wind Speed (km/h)', 'Humidity (%)', 'Precipitation Probability (%)']
for i, (col1, col2) in enumerate(zip(cols, cols[1:])):
    plt.subplot(2, 3, i+1)
    plt.scatter(weather_df[col1], weather_df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300)

# Pair Plot
plt.figure(figsize=(12, 8))
sns.pairplot(weather_df, diag_kind='kde')
plt.tight_layout()
plt.savefig('pair_plot.png', dpi=300)

plt.show()


# In[ ]:

print("-----------------------------------------------------------end of url--------------------------------------------------------------------------------------")


