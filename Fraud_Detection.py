#Importing the required libaries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import pickle
import json
from pickle import dump
from pickle import load

data_columns = None
encode_list = None

with open("fraud_columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
        encode_list = data_columns[10:]

# load the model from disk
#loaded_classifier = load(open('fraud_detection_model.sav', 'rb'))
loaded_classifier = load(open('fraud_detections_model.pickle', 'rb'))

age = st.sidebar.number_input("Enter the Age")
policy_deductable = st.sidebar.number_input("Enter the Policy Deductable Amount")
policy_annual_premium = st.sidebar.number_input("Enter the Policy Annual Premium Amount")
insured_sex = st.sidebar.selectbox('Gender',("FEMALE", "MALE"))
insured_relationship = st.sidebar.selectbox('insured relationship',("husband","not-in-family","other-relative","own-child","unmarried","wife"))
incident_type = st.sidebar.selectbox('incident type',("Multi-vehicle Collision","Single Vehicle Collision","Vehicle Theft","Parked Car"))
collision_type = st.sidebar.selectbox('collision type',("Front Collision","Rear Collision","Side Collision","No info"))
incident_severity = st.sidebar.selectbox('incident severity',("Major Damage","Minor Damage","Total Loss","Trivial Damage"))
authorities_contacted = st.sidebar.selectbox('authorities contacted',("Ambulance","Fire","None","Other","Police"))
number_of_vehicles_involved = st.sidebar.selectbox('number_of_vehicles_involved',('1','2','3','4'))
property_damage = st.sidebar.selectbox('property damage',("Dont Know","NO","YES"))
bodily_injuries = st.sidebar.selectbox('bodily injuries',('0','1','2'))
witnesses = st.sidebar.selectbox('Witnesses',('0','1','2','3'))
police_report_available = st.sidebar.selectbox('police report available',("No","Missing","Yes"))
total_claim_amount = st.sidebar.number_input("Enter the Total Claim Amount")
injury_claim = st.sidebar.number_input("Enter the Injury Claim Amount")
property_claim = st.sidebar.number_input("Enter the Property Claim Amount")
vehicle_claim = st.sidebar.number_input("Enter the Vehicle Claim Amount")

user_input = {'age':age,'policy_deductable':policy_deductable,'policy_annual_premium':policy_annual_premium,'insured_sex':insured_sex,
'insured_relationship':insured_relationship,'incident_type':incident_type,'collision_type':collision_type,'incident_severity':incident_severity,
'authorities_contacted':authorities_contacted,'number_of_vehicles_involved':number_of_vehicles_involved,'property_damage':property_damage,
'bodily_injuries':bodily_injuries,'witnesses':witnesses,'police_report_available':police_report_available,'total_claim_amount':total_claim_amount,
'injury_claim':injury_claim,'property_claim':property_claim,'vehicle_claim':vehicle_claim} 

user_input_df = pd.DataFrame(user_input, index = [0])


# Function to Gets New input/Datapoint
def classify_fraud(age,policy_deductable,policy_annual_premium,insured_sex,insured_relationship,incident_type,collision_type,incident_severity,
                   authorities_contacted,number_of_vehicles_involved,property_damage,bodily_injuries,witnesses,police_report_available,total_claim_amount,
                   injury_claim,property_claim,vehicle_claim):
    data = {'age':age,'policy_deductable':policy_deductable,'policy_annual_premium':policy_annual_premium,'number_of_vehicles_involved':number_of_vehicles_involved,
            'bodily_injuries':bodily_injuries,'witnesses':witnesses,'total_claim_amount':total_claim_amount,'injury_claim':injury_claim,'property_claim':property_claim,
            'vehicle_claim':vehicle_claim} 
    data_df = pd.DataFrame(data, index = [0])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)
    scaled_data_df = pd.DataFrame(data = scaled_data, columns = data_df.columns, index = data_df.index)

    try:
        ins_index = data_columns.index(insured_sex)
    except:
        ins_index = -1

    try:
        inr_index = data_columns.index(insured_relationship)
    except:
        inr_index = -1

    try:
        ict_index = data_columns.index(incident_type)
    except:
        ict_index = -1

    try:
        cot_index = data_columns.index(collision_type)
    except:
        cot_index = -1

    try:
        ics_index = data_columns.index(incident_severity)
    except:
        ics_index = -1

    try:
        auc_index = data_columns.index(authorities_contacted)
    except:
        auc_index = -1

    try:
        prd_index = data_columns.index(property_damage)
    except:
        prd_index = -1

    try:
        por_index = data_columns.index(police_report_available)
    except:
        por_index = -1

    x = np.zeros(len(data_columns))
    x[0] = scaled_data_df.age
    x[1] = scaled_data_df.policy_deductable
    x[2] = scaled_data_df.policy_annual_premium
    x[3] = scaled_data_df.number_of_vehicles_involved
    x[4] = scaled_data_df.bodily_injuries
    x[5] = scaled_data_df.witnesses
    x[6] = scaled_data_df.total_claim_amount
    x[7] = scaled_data_df.injury_claim
    x[8] = scaled_data_df.property_claim
    x[9] = scaled_data_df.vehicle_claim

    if [np.logical_and(ins_index > 0 , ins_index == 0)]:
        x[ins_index] = 1
    if inr_index >= 0:
        x[inr_index] = 1
    if ict_index >= 0:
        x[ict_index] = 1
    if cot_index >= 0:
        x[cot_index] = 1
    if ics_index >= 0:
        x[ics_index] = 1
    if auc_index >= 0:
        x[auc_index] = 1
    if prd_index >= 0:
        x[prd_index] = 1
    if por_index >= 0:
        x[por_index] = 1

    return loaded_classifier.predict([x])[0]

st.subheader('User Input parameters')
st.write(user_input_df)


#Prediction
Classified = None

st.subheader('Detected As')
if st.button("Click Here to Check"):
    Classified = classify_fraud(age,policy_deductable,policy_annual_premium,insured_sex,insured_relationship,incident_type,collision_type,incident_severity,
                   authorities_contacted,number_of_vehicles_involved,property_damage,bodily_injuries,witnesses,police_report_available,total_claim_amount,
                   injury_claim,property_claim,vehicle_claim)
st.success('The Claim is Detected As: {}'.format(Classified))

st.subheader('This Means')
st.write('Claim is Fraud' if Classified == 1 else 'No Fraud')
