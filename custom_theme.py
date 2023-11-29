custom_streamlit_style = """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    #root > div:nth-child(1) > div > div > div > div > \
    section > div {padding-top: 1rem;}

    [data-testid="stSidebar"] { 
        background: linear-gradient(to right, #E3C4FF, #D9ADFF); 
        box-shadow: 15px 15px 10px rgba(0,0,0,0.1);
        border-radius: 10px;

    }   
    [data-testid="stSidebarNav"] {
        background-color: #ffffff;
        font-weight: bold;
        margin-top: -6rem;
        border-style: solid;
        border-width: 20px;
        border-image: linear-gradient(to right, #f5f5f5, #f0f0f0) 1;
    }

    .stSlider {
        background-color: #faf5ff;
        border: 3px solid #c893fc; 
        border-radius: 10px;
        padding: 25px;
        box-shadow: 5px 5px 10px rgba(0,0,0,0.4);
    }
    .stSlider:hover {
        background-color: #ede2ff;
        border: 3px solid #9e70f5; 
    }

    [data-testid="stWidgetLabel"] p {
        font-weight: bold;
        font-size: 16px;
    }

    [data-testid="stDownloadButton"] button {
        border: 3px solid #B3A2E9; 
        background-color: #ECE6FC;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
    }
    [data-testid="stDownloadButton"] button:hover {
        border: 3px solid #8F78CD;
        background-color: #CFC8F2;
    }

    [data-testid="stFormSubmitButton"] button {
        border: 3px solid #B3A2E9; 
        background-color: #ECE6FC;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
    }
    [data-testid="stFormSubmitButton"] button:hover {
        border: 3px solid #8F78CD;
        background-color: #CFC8F2;
    }

    [data-testid="stFileUploader"] {
        border: 3px solid #72B6FC; 
        background-color: #eff6ff;
        padding: 10px;
        border-radius: 10px;
    }
    [data-testid="stFileUploader"]:hover {
        border: 3px solid #4D8FD6;
        background-color: #DCE4F7;
    }

    [data-testid="stCaptionContainer"] {
        textColor: white;
    }
    
    [data-testid="baseButton-header"] {
        background-color: #DCE4F7;
    }
    
    [data-testid="stFileUploader"] {
        border: 3px solid #72B6FC; 
        background-color: #eff6ff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploader"]:hover {
        border: 3px solid #4D8FD6;
        background-color: #DCE4F7;
    }

    [data-testid="stFileUploader"] button {
        border: 3px solid #B3A2E9; 
        background-color: #ECE6FC;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
    }
    [data-testid="stFileUploader"] button:hover {
        border: 3px solid #8F78CD;
        background-color: #CFC8F2;
    }

    [data-testid="stForm"] {
        border: 3px solid #4ac9a2; 
        background-color: #ecfdf5; 
        box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stForm"]:hover {
        border: 3px solid #51B391;
        background-color: #c5f2e1;
    }

    [data-baseweb="select"] {
        font-weight: bold;
        font-size: 16px;
        box-shadow: 5px 5px 4px rgba(0,0,0,0.5);
    }
    [data-testid="stWidgetLabel"] [data-baseweb="textarea"] {
        border: 3px solid #51B391;
        box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
    }
    [data-testid="stExpander"] {
        border: 2px solid #7EB6F6; 
        border-radius: 10px;
        box-shadow: 5px 5px 10px rgba(0,0,0,0.4);
    }
    
    [data-baseweb="select"] [data-baseweb="icon"] {
        border: 3px solid #eff6ff;
        background-color: #eff6ff; 
        box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }

    [data-baseweb="select"] [data-baseweb="icon"]:hover {
        border: 3px solid #d8e3f3;
        background-color: #d8e3f3; 
        box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
    }

    </style>
    """