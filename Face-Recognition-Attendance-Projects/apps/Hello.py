import streamlit as st
import pyodbc
import datetime
# Establishing Connectivity with azure SQL database
server = 'asthasql.database.windows.net'
database = 'sql_facerecog'
username = 'azureadmin'
password = 'asthaface10_'   
driver= '{ODBC Driver 17 for SQL Server}'

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

# function to add data to the azure database
def add_data(input1,input2,input3,input4,input5,input6,input7,input8,input9,input10):
    cursor.execute('INSERT INTO DETAILS VALUES(?,?,?,?,?,?,?,?,?,?)',(input1,input2,input3,input4,input5,input6,input7,input8,input9,input10))
    cnxn.commit()


def app():
    st.title('Hello New User')
    st.header('Add your Details')
    
    
    # Take inputs
    buff, col,= st.columns([6,6])

    input1=buff.text_input('Name',placeholder='Name')

    input2=buff.text_input('Aadhaar Card Number',placeholder='XXXX XXXX XXXX')

    input3=buff.text_input('Driving License Number','')

    input4=buff.date_input("DOB", (datetime.date(1994,1,1)))

    input5=buff.text_input('PAN',placeholder='XXXXXXXXXX')

    input6=col.date_input("Last Covid Test Date",(datetime.date(2016,1,1)))

    input7=col.selectbox("Covid Test Result",('Positive','Negative'),)

    input8=col.selectbox("Gender",('Male','Female','Other'),)

    input9=col.text_input("Email", placeholder='xxxxxx@email.com')

    input10=col.text_input("Mobile Number",placeholder="XXXXXXXXXX")

    a,b,c,= st.columns([6,3,6])

    submitclick=b.button("SUBMIT")
   
    # Insert data to database on clicking submit button and show balloons
    if submitclick:
        add_data(input1,input2,input3,input4,input5,input6,input7,input8,input10,input9)
        st.balloons()



  

            