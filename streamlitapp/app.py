import streamlit as st

import pandas as pd
import joblib

import os



def get_csv(df):
    
    csv = df.to_csv(index=False)
    
    
    return csv
def form_callback():
    st.write(st.session_state.my_option)
    st.write(st.session_state.my_checkbox)
# def listmodel(path:str = "./model"):
    
#     return os.listdir(path)


def st_header(data):
    st.title("Classification App")
    
    with st.container():
        
        col1, col2, col3 = st.columns([1,10,1])
        
        with col2 :
            # st.markdown("{}".format(str(word)))
            uploaded_file = st.file_uploader("Choose a file")
            
            if uploaded_file is not None:

                data = pd.read_csv(uploaded_file)
                
                st.write(data.head())
                
    return data



def st_body():
    # lstmodel = listmodel("./model")
    
    # tmp = [i.split('.')[0] for i in lstmodel]
    col1, col2, col3 = st.columns([1,10,1])
    with col2 :
        with st.form(key='my_form'):
            option = st.selectbox('Select Model:',['GradientBoostingClassifier'],key="my_option")
            
            submitted = st.form_submit_button('Submit')
            if submitted:
                st.write('You selected model: {}'.format(str(option)))
    # return lstmodel[tmp.index(option)]

def st_result(data,clf):
    import swg
    df = None
    if data is not None:

        rs = swg.Swg(data)
        X = rs.scale()
        
        model = joblib.load(f"GradientBoostingClassifier.model")
        z = model.predict(X)
        res = pd.concat([data,pd.DataFrame(z,columns=['Status'])],axis=1)
        col1, col2, col3 = st.columns([1,1,1])
        with col2 :
            st.download_button("Download Classification File",get_csv(res),"result_app.csv")
    
    
def main():
    
    data = None
    data = st_header(data)
    
    clf = st_body()
    st_result(data,clf)

main()
