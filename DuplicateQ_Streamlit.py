from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import streamlit as st
st.title("Duplicate question removel")
upload_file=st.file_uploader("input plz",type="txt")
out_line=[]
if upload_file is not None:
    def common(df):
       q1=set(map((lambda word:word.lower()),df["question1"].split(" ")))
       q2 = set(map((lambda word: word.lower()),df["question2"].split(" ")))
       return len(q1&q2)
    def prediction(t1,t2):
       data={"question1":[t1],"question2":[t2]}
       data=pd.DataFrame(data)
       data["q1_length"]=data["question1"].str.len()
       data["q2_length"]=data["question2"].str.len()
       data["q1_number_of_words"]=data["question1"].apply(lambda row:len(row.split(" ")))
       data["q2_number_of_words"]=data["question2"].apply(lambda row:len(row.split(" ")))
       data["common_words"]=data.apply(common,axis=1)
       question=list(data["question1"])+list(data["question2"])
       new_data=data[["q1_length","q2_length","q1_number_of_words","q2_number_of_words","common_words"]]
    
       cateorical=CountVectorizer(max_features=2000)
       q=cateorical.fit_transform(question)
       q=q.toarray()
       q1,q2=np.vsplit(q,2) 
       q1=pad_sequences(q1, maxlen=2000)
       q2=pad_sequences(q2, maxlen=2000)

       q1=pd.DataFrame(q1,index=data.index)
       q2=pd.DataFrame(q2,index=data.index)

       new_data=pd.concat([q1,q2,new_data],axis=1)

       p=model.predict(new_data)
       return p[0]



    model=load_model(r"Intern\dataset\a.h5")
    with open("Intern\dataset\questions.txt",'r') as f:
       lines=f.readlines()
       for i in range(len(lines)):
         line=lines[i]
         for j in range(i+1,len(lines)):
            t1=line
            t2=lines[j]
            p=prediction(t1,t2)
            if(p>=-3.5):
              out_line.append(t1)
              out_line.append(t2)
            else:
              if(len(t1)==len(t2)):
                if(p>=-4.5):
                   
                   out_line.append(t1)
                   out_line.append(t2)
                else:
                
                  out_line.append(t1)
    with open("Intern\dataset\out.txt",'w') as f:
       for line in out_line:
          f.write(line)               
    st.download_button(
    label="Download Text File",
    data="\n".join(out_line),
    file_name="out.txt",
    mime="text/plain"
    )                              




