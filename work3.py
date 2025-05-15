import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------- Config & Title ----------------------
st.set_page_config(page_title="Financial Forecast Dashboard", layout="wide")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] {
        background-color: #800020 !important;
        color: white !important;
        border-radius: 5px;
        margin-right: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #A52A2A !important;
        color: white !important;
    }X
    .stTabs [aria-selected="true"] {
        background-color: #4B0011 !important;
        color: white !important;
        font-weight: 700;
    }
    .banner {
    /* existing styles preserved */
    display: flex;
    align-items: center;
    justify-content: center;
}
.logo-container {
    margin-right: 20px;
}
.logo-img {
    height: 60px;
    max-width: 300px;
    object-fit: contain;
}
    .main { background-color: #f8f9fa; }
    .banner {
        background-color: #800020;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
    }
    h1.banner-title {
        font-size: 40px;
        font-weight: 900;
        margin: 0;
    }
    .dataframe tbody tr:hover { background-color: #f1f3f5; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
  <div class="logo-container">
        <img src="https://images.ctfassets.net/mviowpldu823/6Kt25rWq7psmWbhC9fvwDZ/120266db4c7fc6fcd36064a736417041/03-Wordmark-watermark__1_.png" width="250">
        
    </div>

 
<div class="banner" style="text-align: center;">
        <h1 class="banner-title" style="margin-top: 10px;">Financial Forecast Dashboard</h1>
</div>
""", unsafe_allow_html=True)


# ---------------------- Sidebar Inputs ----------------------
st.sidebar.markdown("<h2 style='color:#800020;'>📋 Input Parameters</h2>", unsafe_allow_html=True)

initial_intake = st.sidebar.number_input("Initial Fall-1 Intake (students)", value=35)

term_labels = ["Fall-1", "Spring-1", "Fall-2", "Spring-2", "Fall-3", "Spring-3", "Fall-4", "Spring-4", "Fall-5", "Spring-5"]
def_growths = [0.0, 1.0, 100.0, 2.0, 50.0, 10.0, 10.0, 1.0, 1.0, 1.0]
term_growth = [st.sidebar.number_input(f"Growth % {t}", value=def_growths[i]) / 100 for i, t in enumerate(term_labels)]

def_grads = [0, 0, 0, 0, 30, 55, 70, 85, 95, 100]
grad_rates = [st.sidebar.slider(f"Graduation % by {t}", 0, 100, def_grads[i]) / 100 for i, t in enumerate(term_labels)]

ret_early = st.sidebar.slider("Retention T1-T4 (%)", 0, 100, 85) / 100
ret_late = st.sidebar.slider("Retention T5-T10 (%)", 0, 100, 90) / 100

tuition = st.sidebar.number_input("Tuition per Credit ($)", value=1200)
avg_credits = st.sidebar.number_input("Avg Credits per Term", value=6)
credits_per_course = st.sidebar.number_input("Credits per Course", value=3)
faculty_cost = st.sidebar.number_input("Faculty $ per Section", value=12000)
ta_ratio = st.sidebar.number_input("TA:Student Ratio", value=30)
ta_rate = st.sidebar.number_input("TA Hourly Rate ($)", value=25)
ta_hours = st.sidebar.number_input("TA Hours/Week", value=20)
weeks_term = st.sidebar.number_input("Weeks per Term", value=14)
course_dev_cost = st.sidebar.number_input("Course Development Cost ($)", value=50000)
total_courses = st.sidebar.number_input("Total Unique Courses", value=15)

fixed_overhead = st.sidebar.number_input("Fixed Overhead/Term ($)", value=50000)
variable_overhead = st.sidebar.number_input("Variable Overhead per Student ($)", value=50)
cac = st.sidebar.number_input("CAC per New Student ($)", value=7000)
inflation_percent = st.sidebar.slider("Inflation on Costs (Annual) (%)", 0, 100, 7)
inflation = inflation_percent / 100  # Convert to decimal for calculations

# ---------------------- Cohort Calculation ----------------------
num_terms = len(term_labels)
cohort_matrix = np.zeros((num_terms, num_terms))
cohort_matrix[0][0] = round(initial_intake, 2)
for i in range(1, num_terms):
    cohort_matrix[i][i] = round(cohort_matrix[i - 1][i - 1] * (1 + term_growth[i]), 2)

new_intakes = [round(cohort_matrix[i][i], 2) for i in range(num_terms)]

# Retention and Graduation Logic
for cohort in range(num_terms):
    for term in range(cohort + 1, num_terms):
        delta = term - cohort
        retention = ret_early if delta < 4 else ret_late
        retained = cohort_matrix[cohort][term - 1] * retention

        if delta >= 4:
            grads = grad_rates[term] * cohort_matrix[cohort][cohort]
        else:
            grads = 0

        cohort_matrix[cohort][term] = max(0, round(retained - min(retained, grads), 2))

# Graduation by term = grads of cohorts that are eligible
graduated = np.zeros(num_terms)
for term in range(num_terms):
    for cohort in range(term + 1):
        if term - cohort >= 4:
            grads = grad_rates[term] * cohort_matrix[cohort][cohort]
            graduated[term] += grads

graduated = np.round(graduated, 2)

active_students = np.round(cohort_matrix.sum(axis=0), 2)
cohorts = [round(cohort_matrix[i][i], 2) for i in range(num_terms)]

# Cohort Table
df_matrix = pd.DataFrame(np.round(cohort_matrix, 2), columns=term_labels)
df_matrix.insert(0, "Cohort Start Term", term_labels)
df_matrix.insert(1, "Initial Intake (N)", cohorts)
df_matrix.loc["Sum"] = np.round(df_matrix.iloc[:, 2:].sum(numeric_only=True), 2)

# Summary Table
df = pd.DataFrame({
    'Term': term_labels,
    'Active Students': np.round(active_students, 2),
    'Graduated': np.round(graduated, 2)
})




# ---------------------- Revenue Calculation ----------------------
revenue_df = pd.DataFrame({
    'Term': term_labels,
    'Students': active_students,
    'Per Credit': tuition,
    'Credits': avg_credits,
    'Courses' :total_courses,
})
revenue_df['Total Revenue'] = revenue_df['Students'] * tuition * avg_credits

# ---------------------- Cost Calculation ----------------------
cost_df = pd.DataFrame()
cost_df['Term'] = term_labels
cost_df['Faculty'] = revenue_df['Courses'] * faculty_cost
cost_df['TA'] = ((revenue_df['Students'] / ta_ratio).apply(lambda x: int(x + 1)) * ta_rate * ta_hours * weeks_term)
fall1_value = (course_dev_cost * total_courses)/2
cost_df['Course Dev'] = [fall1_value if i == 0 else fall1_value * 0.2 for i in range(len(cost_df))]
cost_df['Variable OH'] = revenue_df['Students'] * variable_overhead
cost_df['Fixed OH'] = fixed_overhead
cost_df['CAC'] = [cohorts[i] * cac for i in range(10)]
cost_df['Total Cost'] = cost_df[['Faculty', 'TA', 'Course Dev', 'Variable OH', 'Fixed OH', 'CAC']].sum(axis=1)
cost_df['With Inflation'] = cost_df['Total Cost'] * [(1 + inflation) ** (i // 2) for i in range(10)]

# ---------------------- Summary Table ----------------------
summary_df = pd.DataFrame({
    'Term': term_labels,
    'Revenue': revenue_df['Total Revenue'],
    'Cost': cost_df['With Inflation'],
})
summary_df['Net'] = summary_df['Revenue'] - summary_df['Cost']
summary_df['Year'] = ['FY2026'] * 2 + ['FY2027'] * 2 + ['FY2028'] * 2 + ['FY2029'] * 2 + ['FY2030'] * 2
yearly_summary = summary_df.groupby('Year').sum(numeric_only=True).reset_index()
yearly_summary['Carried Over'] = yearly_summary['Net'].cumsum()
yearly_summary['Net Margin %'] = (yearly_summary['Net'] / yearly_summary['Revenue']) * 100
yearly_summary['Net Margin %'] = yearly_summary['Net Margin %'].round(2)


# ---------------------- Tabs ----------------------
tabs = st.tabs(["📈 Cohorts", "💵 Revenue", "💰 Cost", "📊 Summary"])

with tabs[0]:
    st.subheader("📈 Cohort Projection")

    # 7. Enrolled Students per Term (New Intake) shown after core dashboards
    st.subheader("📥 New Student Intake per Term")
    intake_df = pd.DataFrame({
    "Term": term_labels,
    "New Intake (Enrolled)": new_intakes
})
    st.dataframe(intake_df.style.format({"New Intake (Enrolled)": "{:.2f}"}).background_gradient(cmap='Purples'))

    # Line chart: Active over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=term_labels, y=active_students, mode='lines+markers', name="Active Students", line=dict(color='#007bff')))
    fig.update_layout(title="Active Students Over Time", xaxis_title="Term", yaxis_title="Count", transition=dict(duration=500, easing='cubic-in-out'))
    st.plotly_chart(fig, use_container_width=True)
    


    # Bar chart: Active vs Graduated per term
    bar_fig = go.Figure(data=[
        go.Bar(name='Active Students', x=term_labels, y=active_students, marker_color='rgb(55, 83, 109)'),
        go.Bar(name='Graduated Students', x=term_labels, y=graduated, marker_color='rgb(26, 118, 255)')
    ])
    bar_fig.update_layout(barmode='group', title="Active vs Graduated Students per Term", xaxis_title="Term", yaxis_title="Count", transition=dict(duration=500, easing='cubic-in-out'))
    st.plotly_chart(bar_fig, use_container_width=True)
   
# Display the Active vs Graduated table with formatting
    st.dataframe(df.style.format({
    "Active Students": "{:.2f}",
    "Graduated": "{:.2f}"
     }).background_gradient(cmap='Blues'))
    
    # Compute diagonal entries (initial intakes by growth)
    for i in range(1, num_terms):
     cohort_matrix[i][i] = round(cohort_matrix[i - 1][i - 1] * (1 + term_growth[i]), 4)

# Fill rest of matrix using Excel-style retention & graduation formulas
    for i in range(num_terms):
     for j in range(i + 1, num_terms):
        delta = j - i
        retention = ret_early if delta < 4 else ret_late
        retained = cohort_matrix[i][j - 1] * retention
        grad_fraction = grad_rates[j] - grad_rates[j - 1] if j > 0 else grad_rates[j]
        grads = grad_fraction * cohort_matrix[i][i]
        cohort_matrix[i][j] = max(0, round(retained - min(retained, grads), 4))

    
# Convert to DataFrame and round
    cohort_df = pd.DataFrame(cohort_matrix, columns=[f"Active T{i+1}" for i in range(num_terms)])
    cohort_df.insert(0, "Initial Intake (N)", [round(cohort_matrix[i][i], 2) for i in range(num_terms)])
    cohort_df.insert(0, "Cohort Start Term", term_labels)
    cohort_df.loc["Sum"] = cohort_df.iloc[:, 2:].sum(numeric_only=True)

# Display the full detailed cohort matrix
    st.subheader("📊 Full Cohort Matrix")
    st.dataframe(cohort_df.style.format(precision=4).background_gradient(cmap="Greens"))    







with tabs[1]:
    st.subheader("💰 Revenue Over Terms")
    revenue_df = pd.DataFrame({
    'Term': term_labels,
    'Students': active_students,
    'Per Credit': [1200] * num_terms,
    'Credits': [6] * num_terms
})
    revenue_df['Total Revenue'] = revenue_df['Students'] * revenue_df['Per Credit'] * revenue_df['Credits']
 
    fig = px.bar(revenue_df, x='Term', y='Total Revenue', color='Term', color_discrete_sequence=px.colors.sequential.Reds)
    fig.update_layout(transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(revenue_df.style.background_gradient(cmap='Greens'))
    

with tabs[2]:
    with st.expander("🧮 Formula Notes"):
        st.markdown("""
        **Cost Formulas:**
        - Faculty = (Students ÷ Credits per Course) × Faculty $ per section
        - TA = ceil(Students ÷ TA Ratio) × Hourly Rate × Weekly Hours × Term Weeks
        - Course Dev = $50,000 if applicable
        - Variable OH = Students × Variable Overhead $
        - Fixed OH = $50,000 (fixed)
        - CAC = New Students × CAC $
        - With Inflation = Total Cost × (1 + inflation)^year_index
        """)
    st.subheader("💰 Cost Breakdown")
    fig = px.bar(cost_df, x='Term', y=['Faculty', 'TA', 'Course Dev', 'Variable OH', 'Fixed OH', 'CAC'], barmode='stack', title="Cost Components", hover_name='Term', color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_layout(transition=dict(duration=500, easing='cubic-in-out'))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cost_df.style.format({
    "Faculty": "$ {:,.2f}",
    "TA": "$ {:,.2f}",
    "Course Dev": "$ {:,.2f}",
    "Variable OH": "$ {:,.2f}",
    "Fixed OH": "$ {:,.2f}",
    "CAC": "$ {:,.2f}",
    "Total Cost": "$ {:,.2f}"
     }).background_gradient(cmap='Blues'))
 

with tabs[3]:
    st.subheader("📊 Summary by Year")
    fig = px.bar(yearly_summary, x='Year', y=['Revenue', 'Cost', 'Net'], barmode='group', color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(transition=dict(duration=500, easing='cubic-in-out'))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(yearly_summary.style.format({
    "Revenue": "$ {:,.2f}",
    "Cost": "$ {:,.2f}",
    "Net": "$ {:,.2f}",
    "Carried Over": "$ {:,.2f}",
    "Net Margin": "$ {:,.2f}",
     }).background_gradient(cmap='Blues'))
