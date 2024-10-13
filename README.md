## **Library Usage Analysis and Visualization**

### **Introduction**

Libraries play a crucial role in community engagement, education, and access to resources. This project aims to analyze and visualize library usage data from the **San Francisco Public Library (SFPL)** to provide insights into patron activity, circulation trends, and demographic patterns.

#### **Objectives:**
- **Analyze library patron behaviors**: Including borrowing patterns, renewals, and circulation activity.
- **Explore relationships**: Between patron age ranges, home libraries, and circulation trends.
- **Forecast library usage**: Using machine learning techniques and advanced regression models.

#### **Data Source Overview:**
- **Dataset**: Library Usage - City and County of San Francisco
- **Source**: San Francisco Open Data
- **Details**:  
  - **Rows**: 450,000 (Each representing a patron record)  
  - **Columns**: 11 (Including total checkouts, renewals, patron type, and circulation activity)  
  - **Last Updated**: March 25, 2024

#### **Technologies Used:**
- **Data Processing**: Python (Pandas, NumPy)  
- **Web Interface**: Streamlit  
- **Machine Learning Models**: Random Forest, ARIMA, LSTM  
- **Regression Models**: Ridge, Lasso, and Elastic Net Regression  
- **Visualization**: Plotly, Matplotlib

---

## **Related Work**

Previous studies on library usage have focused on regional analysis, borrowing patterns, and demand forecasting. This project extends existing research by building an **interactive real-time platform** that integrates predictive analytics to assist in library management and decision-making.

---

## **Key Features**

- **Interactive Visualizations**: Explore checkouts, renewals, and demographic trends.
- **Machine Learning Models**: Predict future usage using ARIMA, LSTM, and advanced regressions.
- **Interactive Map**: Visualize patron activity based on location and filters.
- **Web App**: Accessible through a user-friendly Streamlit interface.

### **App Link**:  
Explore the deployed app here: [Library Usage Analysis App](https://7egjvodv6db5qzqnhot6zc.streamlit.app/)

---

## **Glimpse of Visualizations**

Here are some key visualizations used in the project:

### 1. **ARIMA Forecast of Total Checkouts**  
This chart displays the forecasted total checkouts for future years based on the **ARIMA model**. It demonstrates an upward trend in library usage.

![arima](https://github.com/user-attachments/assets/db1b4070-5e02-4428-8e82-ecc9e2d22cd1)

---

### 2. **Comparison of Actual vs Predicted Values (Ridge, Lasso, ElasticNet)**  
This graph shows a comparison between **actual checkouts** and predictions made by the **Ridge, Lasso, and ElasticNet regression models**, revealing their respective performances.

![lineplot](https://github.com/user-attachments/assets/d26fda29-83f9-4d0a-a66c-6e75844aa5d6)

---

### 3. **Checkouts and Renewals Over the Years**  
This line plot displays the trend of **checkouts and renewals** over the years, offering insights into library usage patterns over time.

![7](https://github.com/user-attachments/assets/7149ff25-2676-4d0b-820e-e5f0e58f3b6a)

---

### 4. **Average Total Checkouts by Home Library Definition**  
This bar chart visualizes the **average total checkouts per library**, helping identify which libraries are most active.

![5](https://github.com/user-attachments/assets/b919366a-82e0-444b-b733-a677390f21cf)

---

5. Interactive Map of Library Usage
This interactive map displays library usage by location, filtered by patron type and year range. It provides valuable insights into geographic patterns in library activity.

![Screenshot (107)](https://github.com/user-attachments/assets/e41a977c-1609-4367-81a4-c6318b43547a)

---

## **Project Structure**

```
Library-Usage-Analysis-and-Visualization-
│
├── app.py                # Streamlit Web Application
├── requirements.txt      # Python Dependencies
├── data/                 # Dataset Folder (Library_Usage_20241011.csv)
├── images/               # Folder Containing Visualizations
│   ├── arima.png
│   ├── lineplot.png
│   ├── 5.png
│   └── 7.png
├── README.md             # Documentation (This File)
```

---

## **How to Run the Project Locally**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NavyaBoga1109/Library-Usage-Analysis-and-Visualization-.git
   cd Library-Usage-Analysis-and-Visualization-
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## **Research Focus Areas**

1. **Factors Influencing Patron Activity**: What drives library patron engagement?
2. **Forecasting Usage Trends**: How can machine learning models forecast future library usage?
3. **Identifying High-Usage Patron Types**: Which patron types (age range, home library) show the highest usage?
4. **Impact of Geographic Location**: How does a patron’s location affect their behavior?
5. **Impact of Notice Preferences**: What role do notice preferences play in engagement?
6. **Circulation Patterns over Time**: How can trend analysis guide resource planning?

---

## **Milestones**

| **Milestone**                  | **Timeline**        |
|--------------------------------|---------------------|
| Data Collection & Cleansing   | Week 1-2            |
| Data Analysis Implementation  | Week 3              |
| Visualization Development     | Week 4-6            |
| Front-End Development         | Week 7-8            |
| Testing & Validation          | Week 9-11           |
| Final Report & Video Demo     | Week 12             |

---

## **Why This Project?**

Libraries benefit from **data-driven insights** to optimize their operations, enhance user experience, and plan for future demands. This project provides a powerful tool for understanding patron behavior and predicting trends, helping libraries make informed decisions.

---

## **Contributing**

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more information.

---

## **Contact**

**Project Author**: Navya Boga  
**GitHub Profile**: [NavyaBoga1109](https://github.com/NavyaBoga1109)

---

### **How to Use This README**

1. **Ensure the images are placed in the `images/` folder** inside the repository.
2. **Copy this README content** into a new `README.md` file.
3. **Commit the changes** and push to GitHub:
   ```bash
   git add README.md images/
   git commit -m "Added README with visualizations"
   git push origin main
   ```
