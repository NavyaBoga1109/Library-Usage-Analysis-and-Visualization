# web.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set up Streamlit layout
st.set_page_config(page_title="Library Usage Analysis", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Go to", 
    ["ABOUT DATA", "VISUALIZATIONS", "MACHINE LEARNING MODELS", 
     "PREDICTING TRENDS (ARIMA AND LSTM)", "INTERACTIVE MAP"]
)

# Use relative path since the file is now in the 'data' folder within the repository
file_path = "data/Library_Usage_20241011.csv"
output_directory = "data"
output_file = os.path.join(output_directory, "Library_Usage_Cleaned.csv")

# Load and clean the dataset
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(file_path, low_memory=False)

    # Handle missing values
    df['Total Checkouts'] = df['Total Checkouts'].fillna(df['Total Checkouts'].mean())
    df['Total Renewals'] = df['Total Renewals'].fillna(df['Total Renewals'].mean())
    df.dropna(subset=['Age Range', 'Home Library Definition'], inplace=True)

    # Normalize date formats
    df['Circulation Active Year'] = pd.to_numeric(df['Circulation Active Year'], errors='coerce')

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Detect and remove outliers using IQR method
    Q1 = df[['Total Checkouts', 'Total Renewals']].quantile(0.25)
    Q3 = df[['Total Checkouts', 'Total Renewals']].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[['Total Checkouts', 'Total Renewals']] < (Q1 - 1.5 * IQR)) | 
              (df[['Total Checkouts', 'Total Renewals']] > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Save cleaned data
    df.to_csv(output_file, index=False)

    return df

# Load data
df = load_and_clean_data()

# Function to display heatmap
def show_heatmap(description, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(description, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)

# Page 1: ABOUT DATA
if page == "ABOUT DATA":
    st.header("About the Data")
    st.write("""
        This project analyzes the San Francisco Public Library (SFPL) usage data to uncover key trends 
        in patron activity and predict future circulation patterns using machine learning models.
    """)

    # Display original and cleaned data descriptions
    st.subheader("Original Data Description")
    original_description = pd.read_csv(file_path, low_memory=False).describe()
    st.write(original_description)
    show_heatmap(original_description, "Original Data Description (Heatmap)")

    st.subheader("Cleaned Data Description")
    cleaned_description = df.describe()
    st.write(cleaned_description)
    show_heatmap(cleaned_description, "Cleaned Data Description (Heatmap)")

    # Display first 5 rows of original and cleaned data
    st.subheader("First 5 Rows of Original Data")
    st.write(pd.read_csv(file_path, low_memory=False).head())

    st.subheader("First 5 Rows of Cleaned Data")
    st.write(df.head())

# Page 2: VISUALIZATIONS (EDA Integration)
if page == "VISUALIZATIONS":
    st.header("Exploratory Data Analysis (EDA) Visualizations")

    # Box Plot: Total Checkouts by Age Range
    st.subheader("Total Checkouts by Age Range")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Age Range', y='Total Checkouts', ax=ax)
    ax.set_title('Total Checkouts by Age Range')
    ax.set_xlabel('Age Range')
    ax.set_ylabel('Total Checkouts')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write("""
    **Insight:** This box plot shows the distribution of total checkouts across different age ranges. 
    It highlights the spread and identifies potential outliers within each age category.
    """)

    # Line Plot: Checkouts and Renewals by Circulation Active Year
    st.subheader("Checkouts and Renewals Over the Years")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='Circulation Active Year', y='Total Checkouts', label='Total Checkouts', ax=ax)
    sns.lineplot(data=df, x='Circulation Active Year', y='Total Renewals', label='Total Renewals', ax=ax)
    ax.set_title('Checkouts and Renewals Over the Years')
    ax.set_xlabel('Circulation Active Year')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)
    st.write("""
    **Insight:** This line plot reveals trends in library activity over the years. A comparison between 
    total checkouts and renewals can indicate shifts in borrowing behavior.
    """)

    # Pie Chart: Distribution of Patron Types
    st.subheader("Distribution of Patron Types")
    patron_type_counts = df['Patron Type Definition'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(patron_type_counts, labels=patron_type_counts.index, autopct='%1.1f%%', startangle=140, 
           colors=sns.color_palette("pastel"))
    ax.set_title('Distribution of Patron Types')
    ax.axis('equal')
    st.pyplot(fig)
    st.write("""
    **Insight:** This pie chart provides a quick view of the proportion of different patron types. 
    It helps to understand the demographic distribution of library users.
    """)

    # Scatter Plot: Total Checkouts vs Total Renewals
    st.subheader("Scatter Plot of Total Checkouts vs Total Renewals")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Total Checkouts', y='Total Renewals', ax=ax)
    ax.set_title('Scatter Plot of Total Checkouts vs Total Renewals')
    ax.set_xlabel('Total Checkouts')
    ax.set_ylabel('Total Renewals')
    st.pyplot(fig)
    st.write("""
    **Insight:** This scatter plot shows the relationship between total checkouts and renewals. 
    It helps identify correlations or patterns between these two metrics.
    """)

    # Histogram for Total Checkouts
    st.subheader("Distribution of Total Checkouts")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df['Total Checkouts'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Total Checkouts')
    ax.set_xlabel('Total Checkouts')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("""
    **Insight:** The histogram displays the frequency distribution of total checkouts. 
    The density curve (KDE) provides an estimate of the probability distribution.
    """)

    # Countplot for Age Range
    st.subheader("Age Range Distribution of Patrons")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Age Range', ax=ax)
    ax.set_title('Age Range Distribution of Patrons')
    ax.set_xlabel('Age Range')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write("""
    **Insight:** This count plot indicates how library usage is distributed across different age groups, 
    providing insights into the demographics of the patrons.
    """)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df[['Total Checkouts', 'Total Renewals', 'Circulation Active Year', 'Year Patron Registered']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    st.write("""
    **Insight:** The correlation matrix highlights relationships between numerical variables. 
    Values closer to 1 or -1 indicate strong positive or negative correlations.
    """)

    # Bar Plot: Average Total Checkouts by Patron Type
    st.subheader("Average Total Checkouts by Patron Type")
    grouped_data = df.groupby('Patron Type Definition')['Total Checkouts'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=grouped_data, x='Patron Type Definition', y='Total Checkouts', ax=ax)
    ax.set_title('Average Total Checkouts by Patron Type')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write("""
    **Insight:** This bar plot displays the average number of total checkouts per patron type. 
    It helps identify which patron groups are the most active borrowers. 
    High values indicate frequent library usage within certain patron types.
    """)

    # Bar Plot: Average Total Checkouts by Home Library Definition
    st.subheader("Average Total Checkouts by Home Library Definition")
    grouped_home_library = df.groupby('Home Library Definition')['Total Checkouts'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=grouped_home_library, x='Home Library Definition', y='Total Checkouts', ax=ax)
    ax.set_title('Average Total Checkouts by Home Library')
    plt.xticks(rotation=90)
    st.pyplot(fig)
    st.write("""
    **Insight:** This bar plot reveals the average number of total checkouts across different home libraries. 
    It provides insight into which libraries have the most engaged users. 
    The comparison helps libraries understand their usage patterns and benchmark against others.
    """)

# Page 3: Machine Learning Models Section
if page == "MACHINE LEARNING MODELS":
    st.header("Machine Learning Models")

    # Ensure all necessary imports are included at the top of your script
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    # Data Preprocessing for ML Models
    df.replace(['False', 'Null', '', None], pd.NA, inplace=True)
    df = pd.get_dummies(df, columns=['Patron Type Definition', 'Home Library Definition', 'Age Range'], drop_first=True)
    df.dropna(inplace=True)

    # Define Features and Target Variable
    X = df[['Total Renewals', 'Year Patron Registered', 'Circulation Active Year'] + 
           [col for col in df.columns if 'Patron Type Definition' in col or 
            'Home Library Definition' in col or 'Age Range' in col]]
    y = df['Total Checkouts']

    # Split Data into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Linear Regression
    st.subheader("Linear Regression")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_lr = linear_model.predict(X_test)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    st.write(f"Mean Squared Error (MSE): {mse_lr}")
    st.write(f"R-squared (R2): {r2_lr}")

    # Plot: Actual vs Predicted - Linear Regression
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_lr, ax=ax)
    ax.set_title('Actual vs Predicted Total Checkouts - Linear Regression')
    ax.set_xlabel('Actual Total Checkouts')
    ax.set_ylabel('Predicted Total Checkouts')
    st.pyplot(fig)

    # Feature Importance - Linear Regression
    linear_feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': linear_model.coef_
    }).sort_values(by='Importance', ascending=False)

    st.write("Feature Importance - Linear Regression")
    st.dataframe(linear_feature_importance)

    # Ridge, Lasso, and Elastic Net Models
    st.subheader("Ridge, Lasso, and Elastic Net Regression Models")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ridge Regression
    ridge_model = Ridge()
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # Lasso Regression
    lasso_model = Lasso()
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # Elastic Net Regression
    elastic_net_model = ElasticNet()
    elastic_net_model.fit(X_train_scaled, y_train)
    y_pred_elastic_net = elastic_net_model.predict(X_test_scaled)
    mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)
    r2_elastic_net = r2_score(y_test, y_pred_elastic_net)

    # Display Model Results
    st.write(f"Ridge Regression - MSE: {mse_ridge}, R2: {r2_ridge}")
    st.write(f"Lasso Regression - MSE: {mse_lasso}, R2: {r2_lasso}")
    st.write(f"Elastic Net Regression - MSE: {mse_elastic_net}, R2: {r2_elastic_net}")

    # Comparison Plot - Actual vs Predicted (All Models)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label='Actual', color='black', linestyle='dashed', linewidth=2)
    ax.plot(y_pred_ridge, label='Ridge', color='blue')
    ax.plot(y_pred_lasso, label='Lasso', color='green')
    ax.plot(y_pred_elastic_net, label='Elastic Net', color='red')
    ax.set_title('Comparison of Actual vs Predicted Values')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Total Checkouts')
    ax.legend()
    st.pyplot(fig)

    # Function to Plot Heatmaps
    def plot_heatmap(data, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.set_index('Feature').T, annot=True, cmap='coolwarm', cbar=False, linewidths=.5, fmt=".2f", ax=ax)
        ax.set_title(title)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # Heatmaps for Feature Importance
    plot_heatmap(pd.DataFrame({'Feature': X.columns, 'Importance': ridge_model.coef_}).sort_values(by='Importance', ascending=False), 
                 "Ridge Regression - Feature Importance")
    plot_heatmap(pd.DataFrame({'Feature': X.columns, 'Importance': lasso_model.coef_}).sort_values(by='Importance', ascending=False), 
                 "Lasso Regression - Feature Importance")
    plot_heatmap(pd.DataFrame({'Feature': X.columns, 'Importance': elastic_net_model.coef_}).sort_values(by='Importance', ascending=False), 
                 "Elastic Net Regression - Feature Importance")

    # Residual Plots
    st.subheader("Residual Plots")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=range(len(y_test)), y=y_test - y_pred_ridge, label='Ridge Residuals', color='blue', ax=ax)
    sns.lineplot(x=range(len(y_test)), y=y_test - y_pred_lasso, label='Lasso Residuals', color='green', ax=ax)
    sns.lineplot(x=range(len(y_test)), y=y_test - y_pred_elastic_net, label='Elastic Net Residuals', color='red', ax=ax)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title('Residual Plot: Actual vs Predicted Differences')
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.legend()
    st.pyplot(fig)

# Page 4: PREDICTING TRENDS (ARIMA AND LSTM)
if page == "PREDICTING TRENDS (ARIMA AND LSTM)":
    st.header("Predicting Trends: ARIMA and LSTM")

    # Group data by year
    data = df.groupby(df['Circulation Active Year'])['Total Checkouts'].sum()

    # --- ARIMA Model ---
    st.subheader("ARIMA Model Predictions")
    arima_model = ARIMA(data, order=(1, 1, 1))
    arima_result = arima_model.fit()
    forecast = arima_result.forecast(steps=5)
    forecast_years = np.arange(data.index[-1] + 1, data.index[-1] + 6)

    # Display ARIMA forecast as a DataFrame
    arima_output = pd.DataFrame({'Year': forecast_years, 'ARIMA Forecast': forecast.values})
    st.write(arima_output)

    # Plot ARIMA Forecast
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data.index, data, label='Actual Total Checkouts', marker='o')
    ax.plot(forecast_years, forecast, label='ARIMA Forecast', marker='o')
    ax.set_title('ARIMA Forecast of Total Checkouts')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Checkouts')
    ax.legend()
    st.pyplot(fig)

    # --- LSTM Model ---
    st.subheader("LSTM Model Predictions")

    # Preprocessing for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Prepare the training data
    X_train, y_train = [], []
    for i in range(3, len(scaled_data)):
        X_train.append(scaled_data[i-3:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # LSTM Forecast
    last_3_steps = scaled_data[-3:]
    last_3_steps = last_3_steps.reshape((1, last_3_steps.shape[0], 1))
    lstm_forecast = model.predict(last_3_steps)
    lstm_forecast_value = scaler.inverse_transform(lstm_forecast)[0][0]

    # Display LSTM forecast as a DataFrame
    lstm_output = pd.DataFrame({'Year': [data.index[-1] + 1], 'LSTM Forecast': [lstm_forecast_value]})
    st.write(lstm_output)

    # Plot LSTM Forecast
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data.index, data, label='Actual Total Checkouts', marker='o')
    ax.plot([data.index[-1] + 1], lstm_forecast_value, 'ro', label='LSTM Forecast')
    ax.set_title('LSTM Forecast of Total Checkouts')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Checkouts')
    ax.legend()
    st.pyplot(fig)

    # Combine ARIMA and LSTM outputs into one table
    combined_output = pd.concat([arima_output, lstm_output], axis=0, ignore_index=True)
    st.subheader("Combined Forecast Outputs")
    st.write(combined_output)


# Page 5: INTERACTIVE MAP
if page == "INTERACTIVE MAP":
    st.header("Interactive Map of Library Usage")

    # Group the data by 'Home Library Definition' and aggregate
    library_data = df.groupby('Home Library Definition').agg({
        'Total Checkouts': 'sum',
        'Total Renewals': 'sum',
        'Age Range': lambda x: x.mode()[0],  # Most frequent age group
        'Patron Type Definition': lambda x: x.mode()[0]  # Most common patron type
    }).reset_index()

    # Add latitude and longitude columns (mock data for demonstration)
    library_data['Latitude'] = np.random.uniform(37.7, 37.8, len(library_data))  # Example: SF latitudes
    library_data['Longitude'] = np.random.uniform(-122.5, -122.4, len(library_data))  # Example: SF longitudes

    # Sidebar filters: Patron type and year range selection
    st.sidebar.subheader("Filters")
    selected_patron_type = st.sidebar.selectbox(
        "Select Patron Type", df['Patron Type Definition'].unique()
    )
    selected_year_range = st.sidebar.slider(
        "Select Year Range", 
        int(df["Circulation Active Year"].min()), 
        int(df["Circulation Active Year"].max()), 
        (2004, 2023)
    )

    # Filter data based on the selected patron type and year range
    filtered_data = df[
        (df['Patron Type Definition'] == selected_patron_type) &
        (df['Circulation Active Year'].between(selected_year_range[0], selected_year_range[1]))
    ]

    # Group the filtered data by 'Home Library Definition'
    filtered_library_data = filtered_data.groupby('Home Library Definition').agg({
        'Total Checkouts': 'sum',
        'Total Renewals': 'sum',
        'Age Range': lambda x: x.mode()[0]  # Most common age group
    }).reset_index()

    # Add latitude and longitude to filtered data
    filtered_library_data = filtered_library_data.merge(
        library_data[['Home Library Definition', 'Latitude', 'Longitude']], 
        on='Home Library Definition', 
        how='left'
    )

    # Map: Plot the filtered library usage data
    fig = px.scatter_mapbox(
        filtered_library_data,
        lat="Latitude",
        lon="Longitude",
        size="Total Checkouts",
        color="Total Checkouts",
        hover_name="Home Library Definition",
        hover_data=["Total Renewals", "Age Range"],
        color_continuous_scale=px.colors.sequential.Oranges,
        size_max=15,
        zoom=10,
        mapbox_style="carto-positron"
    )

    # Add title to the map
    fig.update_layout(
        title="Library Usage Map: Filtered by Patron Type and Year Range",
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )

    # Display the interactive map in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Summary insights based on the filtered data
    st.subheader("Summary Insights")
    total_checkouts = filtered_library_data['Total Checkouts'].sum()
    most_common_library = filtered_library_data.loc[filtered_library_data['Total Checkouts'].idxmax()]['Home Library Definition']
    st.write(f"Total Checkouts (Filtered): **{total_checkouts}**")
    st.write(f"Most Active Library: **{most_common_library}**")
    st.write(f"Filtered by Patron Type: **{selected_patron_type}** from **{selected_year_range[0]}** to **{selected_year_range[1]}**.")

    # Optional: Allow comparison between two libraries
    st.subheader("Library Comparison")
    library1 = st.selectbox("Select Library 1", library_data['Home Library Definition'].unique())
    library2 = st.selectbox("Select Library 2", library_data['Home Library Definition'].unique())

    # Extract data for the selected libraries
    data_library1 = library_data[library_data['Home Library Definition'] == library1]
    data_library2 = library_data[library_data['Home Library Definition'] == library2]

    # Display comparison data
    st.write(f"**{library1}** - Total Checkouts: {data_library1['Total Checkouts'].values[0]}, Total Renewals: {data_library1['Total Renewals'].values[0]}")
    st.write(f"**{library2}** - Total Checkouts: {data_library2['Total Checkouts'].values[0]}, Total Renewals: {data_library2['Total Renewals'].values[0]}")

    # Optional: Plot comparison between the two libraries
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(library1, data_library1['Total Checkouts'].values[0], label=f'{library1} Checkouts')
    ax.bar(library2, data_library2['Total Checkouts'].values[0], label=f'{library2} Checkouts')
    ax.set_title('Library Checkouts Comparison')
    ax.legend()
    st.pyplot(fig)

