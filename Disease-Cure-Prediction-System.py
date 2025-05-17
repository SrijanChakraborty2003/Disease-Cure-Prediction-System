import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
import joblib
import torch
st.title("Disease & Cure Prediction System")

@st.cache_resource
def load_model():
  with open("xgb_model.pkl",'rb') as f:
     return pickle.load(f)
@st.cache_data
def load_table():
  return pd.read_csv("User_input_table.csv")
@st.cache_data
def load_data():
  return pd.read_csv("Disease_Data-set.csv")
@st.cache_resource
def load_SentenceTransformer():
  return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_resource
def load_label_encoder():
  return joblib.load('label_encoder.pkl')
symptoms = ['pain during urination', 'abnormal discharge', 'intermenstrual bleeding',
           'abdominal pain', 'abnormal bleeding', 'pelvic pain', 'pain during intercourse',
           'vaginal discharge', 'barking cough', 'runny nose', 'fever', 'stridor', 'rash', 'joint pain',
           'severe headache', 'high fever', 'loss of awareness', 'confusion', 'uncontrollable jerking movements',
           'seizures', 'sore throat', 'painful urination', 'jaundice', 'loss of appetite', 'fatigue', 'knee pain', 'swelling',
           'hip pain', 'stiffness', 'swollen lymph nodes', 'red eyes', 'itching', 'visible lice or eggs', 'sores on the scalp', 'irritability',
           'headache', 'sensitivity to light', 'stiff neck', 'stomach cramps', 'vomiting', 'nausea', 'diarrhea', 'urinary urgency', 'quickly feeling full',
           'abdominal bloating', 'unusual discharge', 'pain when standing or on first steps', 'heel pain', 'chills', 'agitation', 'hydrophobia', 'severe diarrhea',
           'strawberry tongue', 'red rash', 'pain crises', 'frequent infections', 'anemia', 'swelling in hands/feet', 'muscle stiffness', 'lockjaw', 'difficulty swallowing',
           'spasms', 'red or skin-colored welts', 'achiness', 'skin changes', 'swollen veins', 'severe cough', 'exhaustion', 'mild fever', 'conjunctivitis', 'weakness',
           'palpitations', 'dizziness', 'shortness of breath', 'difficulty starting urination', 'weak urinary stream', 'frequent urination', 'muscle pain',
           'cognitive difficulties', 'sleep problems', 'persistent fatigue', 'redness', 'heat in the affected joint', 'blurred vision', 'fainting',
           'daytime fatigue', 'waking up during night', 'difficulty falling asleep', 'loss of flexibility', 'cold sweat', 'chest pain',
           'nasal obstruction', 'bloody nose', 'hearing loss', 'ear pain', 'restless sleep', 'daytime sleepiness',
           'gasping during sleep', 'loud snoring', 'bloating', 'painful joints', 'heart symptoms', 'skin rashes',
           'weight loss', 'burning sensation', 'soreness', 'chronic vulvar pain', 'discomfort', 'coordination problems', 'memory issues',
           'nosebleeds', 'increased thirst', 'extreme hunger', 'chest tightness', 'wheezing', 'coughing', 'heart palpitations', 'breathing difficulties',
           'frequent respiratory infections', 'chronic cough', 'cough', 'body aches', 'reduced mobility', 'persistent sadness', 'sleep disturbances', 'loss of interest',
           'visual disturbances', 'sensitivity to light/sound', 'cold intolerance', 'weight gain', 'dry skin', 'depression', 'balance problems', 'slowed movement', 'tremors',
           'coughing blood', 'night sweats', 'recurring fever', 'rapid weight loss', 'scaly patches', 'heartburn', 'regurgitation', 'delusions', 'hallucinations',
           'disorganized thinking', 'emotional flatness', 'salty skin', 'digestive issues', 'persistent cough', 'frequent lung infections', 'restlessness',
           'excessive worry', 'difficulty concentrating', 'low blood pressure', 'skin darkening', 'fractures', 'back pain', 'stooped posture',
           'loss of height', 'changes in urination', 'widespread pain', 'swallowing difficulties', 'neck pain', 'hoarseness', 'lump in neck',
           'excessive bleeding', 'easy bruising', 'rapid heartbeat', 'nervousness', 'increased sweating', 'scarring', 'pimples', 'blackheads',
           'oily skin', 'mood swings', 'mania', 'anxiety', 'menstrual irregularity', 'indigestion', 'sudden pain in the upper right abdomen',
           'decreased libido', 'odor', 'facial weakness', 'twitching', 'difficulty closing an eye', 'drooping on one side of the face', 'uneven shoulders',
           'difficulty breathing', 'fading colors', 'night vision problems', 'extreme weight loss', 'fear of gaining weight', 'facial pain', 'nasal congestion',
           'reduced sense of smell', 'pain in the right upper abdomen', 'irregular periods', 'hot flashes', 'mood changes', 'sudden shortness of breath',
           'cold fingers/toes', 'color changes in skin', 'numbness', 'rapid heart rate', 'sensitivity to touch', 'flu-like symptoms', 'painful rash',
           'burning pain in stomach', 'leg pain during exercise', 'sores or wounds on toes/legs', 'weak pulses in legs', 'sudden chest pain', 'cyanosis',
           "raynaud's phenomenon", 'skin thickening', 'dark urine', 'ringing in the ears', 'muscle weakness', 'double vision', 'memory loss', 'disorientation',
           'difficulty speaking', 'slow movement', 'loss of taste or smell', 'muscle aches', 'sweating', 'red patches on skin', 'scaly skin', 'dry cracked skin',
           'red patches', 'inflammation', 'whiteheads', 'cough with phlegm', 'chest discomfort', 'mucus production', 'upper abdominal pain', 'severe flank pain',
           'blood in urine', 'gas', 'diarrhea or constipation', 'light sensitivity', 'aura', 'trouble speaking', 'loss of balance', 'sudden numbness', 'vision problems',
           'tingling', 'pale skin', 'frequent urge', 'cloudy urine', 'burning urination', 'excess hair', 'acne', 'pain during bowel movements', 'rectal bleeding'
           ]
@st.cache_data
def load_symptom_embeddings():
  embeddings = model.encode(symptoms)
  return np.array(embeddings)
xgb_model=load_model()
us_df=load_table()
disease_dataset_df=load_data()
le=load_label_encoder()
model=load_SentenceTransformer()
symp_list_embedding = torch.from_numpy(load_symptom_embeddings())
if 'gender' not in st.session_state:
    st.session_state.gender = None
def tokenizer(inp):
  inp = inp.lower()
  raw_tokens = inp.split(" ")
  cleaned_tokens = []
  for i in raw_tokens:
      w = "".join([j for j in i if j.isalnum()])
      cleaned_tokens.append(w)
  if st.session_state.gender is None:
      if "male" in cleaned_tokens:
          st.session_state.gender = "male"
      elif "female" in cleaned_tokens:
          st.session_state.gender = "female"
  if st.session_state.gender is None:
      return None
  user_symp_tokens=[]
  for i in range(1,8):
    if(i==7):
      for j in range (len(cleaned_tokens)-6):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]+" "+cleaned_tokens[j+2]+" "+cleaned_tokens[j+3]+" "+cleaned_tokens[j+4]+" "+cleaned_tokens[j+5]+" "+cleaned_tokens[j+6]
        if(token not in user_symp_tokens):
            user_symp_tokens.append(token)

    if(i==6):
      for j in range (len(cleaned_tokens)-5):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]+" "+cleaned_tokens[j+2]+" "+cleaned_tokens[j+3]+" "+cleaned_tokens[j+4]+" "+cleaned_tokens[j+5]
        if(token not in user_symp_tokens):
            user_symp_tokens.append(token)

    if(i==5):
      for j in range (len(cleaned_tokens)-4):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]+" "+cleaned_tokens[j+2]+" "+cleaned_tokens[j+3]+" "+cleaned_tokens[j+4]
        if(token not in user_symp_tokens):
            user_symp_tokens.append(token)

    if(i==4):
      for j in range (len(cleaned_tokens)-3):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]+" "+cleaned_tokens[j+2]+" "+cleaned_tokens[j+3]
        if(token not in user_symp_tokens):
            user_symp_tokens.append(token)

    if(i==3):
      for j in range(len(cleaned_tokens)-2):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]+" "+cleaned_tokens[j+2]
        if(token not in user_symp_tokens):
          user_symp_tokens.append(token)

    if(i==2):
      for j in range(len(cleaned_tokens)-1):
        token=cleaned_tokens[j+0]+" "+cleaned_tokens[j+1]
        if(token not in user_symp_tokens):
          user_symp_tokens.append(token)

    if(i==1):
      for j in range(len(cleaned_tokens)):
        token=cleaned_tokens[j]
        if(token not in user_symp_tokens):
          user_symp_tokens.append(token)
  user_symp_tokens_embeddings=model.encode(user_symp_tokens, convert_to_tensor=True)
  user_symptoms=[]
  for i in (user_symp_tokens_embeddings):
    cos_score=util.cos_sim(i,symp_list_embedding)
    b=cos_score.argmax()
    mapped_symptom=symptoms[b]
    score=cos_score[0][b].item()
    if score>0.70:
      if mapped_symptom not in user_symptoms:
          user_symptoms.append(mapped_symptom)
  gender="Gender_"+gender
  user_symptoms.append(gender)
  return user_symptoms
chat_container = st.container()
user_symptoms=[]
inp=st.chat_input("Enter your symptoms...")
if inp:
    user_symptoms=tokenizer(inp)
    if user_symptoms is None:
        st.session_state.gender = st.selectbox("Please select your gender:", ["male", "female"])
        st.warning("Please select your gender to continue.")
    else:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(inp)
        for i in user_symptoms:
            if i=="Gender_male":
                us_df[i]=1.0
            elif i=="Gender_female":
                us_df[i]=1.0
            else:
                us_df[i]=1
    test=xgb_model.predict(us_df)
    predicted_disease=le.inverse_transform(test)[0]
    cure = disease_dataset_df.loc[disease_dataset_df['Disease'] == predicted_disease, 'Cure'].values[0]
    with st.chat_message("assistant"):
      st.markdown(f"Disease: {predicted_disease}  \nCure: {cure}")
for var in ["user_symptoms", "gender", "inp", "us_df", "predicted_disease","test","user_symp_tokens_embeddings","token","user_symp_tokens","raw_tokens","w"]:
    if var in st.session_state:
        del st.session_state[var]
