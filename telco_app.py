import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load & process the dataset
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("E:\Telecom_Data_Churn_Prediction\Dataset\Telco-Customer-Churn.csv")
    df1=df.copy()
    df1.drop(['customerID'],axis=1,inplace=True)
    X=df1.drop('Churn',axis=1)
    y=df1['Churn']
    X['StreamingServices']=X['StreamingTV']+X['StreamingMovies']
    X['StreamingServices']=X['StreamingServices'].replace({'NoNo': 'NoneStreamingServices', 
                                                             'YesNo': 'OnlyStreamingTV', 
                                                             'NoYes': 'OnlyStreamingMovies','YesYes': 'BothStreamingServices'})
    X.drop(['StreamingTV','StreamingMovies'],inplace=True,axis=1)
    X['InternetServices']=X.apply(lambda row:'DSL Only' if row['InternetService']=='DSL' and row['OnlineBackup']=='No' else
                                          'Fiber Optic Only' if row['InternetService']=='Fiber optic' and row['OnlineBackup']=='No' else
                                          'Internet and Backup' if (row['InternetService']=='DSL' or row['InternetService']=='Fiber optic') and row['OnlineBackup'] == 'Yes' else
                                          'No Internet Service',axis=1)
    X=X.drop(['InternetService','OnlineBackup'],axis=1)
    X['SecurityServices']=X['OnlineSecurity']+X['DeviceProtection']
    #Processing new column values
    X['SecurityServices']=X['SecurityServices'].replace({'NoNo': 'NoneSecurityServices',
                                                           'YesNo': 'OnlyOnlineSecurity',
                                                           'NoYes': 'OnlyDeviceProtection','YesYes': 'BothSecurityServices'})
    X=X.drop(['OnlineSecurity','DeviceProtection'],axis=1)
    columns=['TechSupport','Contract','PaymentMethod','SecurityServices','StreamingServices','InternetServices']
    one_hot=OneHotEncoder()
    encode_cols=one_hot.fit_transform(X[columns])
    # creating new dataframe with encoded values
    encoded_df=pd.DataFrame(encode_cols.toarray(),columns=one_hot.get_feature_names_out(columns))
    # drop original columns from dataframe
    X.drop(columns,inplace=True,axis=1)
    # concatenate new columns in original dataframe
    X=pd.concat([df,encoded_df],axis=1)
    X=X[['tenure', 'MonthlyCharges', 'TechSupport_No', 'Contract_Month-to-month',
       'Contract_Two year', 'PaymentMethod_Electronic check',
       'SecurityServices_NoneSecurityServices',
       'SecurityServices_OnlyDeviceProtection',
       'StreamingServices_BothStreamingServices',
       'InternetServices_Fiber Optic Only']]
    return X

df=load_and_process_data()

df2=pd.read_csv("E:\Telecom_Data_Churn_Prediction\Dataset\Telco-Customer-Churn.csv")

# Preprocess the dataset
X=df2.drop("Churn", axis=1)
y=df2["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model=joblib.load("E:\Telecom_Data_Churn_Prediction\model_gb_selected_features.pkl")
# Display the header
st.subheader("Telco Data Churn Dataset")
# Allow user input for prediction
st.subheader("Customer Churn Prediction")
st.subheader("Here for most categorical columns , 0.0 means 'No' and 1.0 means 'Yes'")
customer_data = {}
customer_data["tenure"] = st.number_input("Tenure", min_value=0, step=1)
customer_data["MonthlyCharges"] = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
customer_data["TechSupport_No"] = st.selectbox("No Tech Support", options=df["TechSupport_No"].unique())
customer_data["Contract_Month-to-month"] = st.selectbox("Is Contract renewable after 1 month", options=df["Contract_Month-to-month"].unique())
customer_data["Contract_Two year"] = st.selectbox("Is Contract renewable after 2 years", options=df["Contract_Two year"].unique())
customer_data["PaymentMethod_Electronic check"] = st.selectbox("Is payment via electronic check", options=df["PaymentMethod_Electronic check"].unique())
customer_data["SecurityServices_NoneSecurityServices"] = st.selectbox("No Security Service", options=df["SecurityServices_NoneSecurityServices"].unique())
customer_data["SecurityServices_OnlyDeviceProtection"] = st.selectbox("Is device protection service availed", options=df["SecurityServices_OnlyDeviceProtection"].unique())
customer_data["StreamingServices_BothStreamingServices"] = st.selectbox("Are both TV / movie streaming services availed", options=df["StreamingServices_BothStreamingServices"].unique())
customer_data["InternetServices_Fiber Optic Only"] = st.selectbox("Is only fiber optic service availed", options=df["InternetServices_Fiber Optic Only"].unique())
customer_df = pd.DataFrame([customer_data])

if st.button("Predict Churn"):
    prediction = model.predict(customer_df)
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    st.write("The customer is predicted to be:", churn_status)