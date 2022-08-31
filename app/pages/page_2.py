import streamlit as st
import os
import pandas as pd
import plotly.express as px



from utilsstreamlit import whole_network_plotter, read_plot_info,display_grid


def run_app():

    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]

    st.title('Syntax networks plotter')

    iol = st.radio('Would you like to show data of inflected or lemmatized forms:',
        options = ['inflected','lemmatized'])
    allorfew = st.radio('Would you like to show all words or just a few?:',
        options = ['All','Custom'])
    if allorfew == 'All':
        nwords = 0
    else:
        nwords = st.text_input('Input desired number of words')
        nwords = int(nwords)
    noc = st.radio('How many clusters would you like to visualizw:',
        options = [2,3,4,5])
    nlang = st.radio('Would you like two show one or several languages?:',
        options = ['One','Several'])
    if nlang == 'One':
        lang = st.selectbox('Pick a language:',
            (lang for lang in languagelist))
        df = read_plot_info(lang,nwords,iol)
        
    else:
        dflang = pd.DataFrame()
        dflang['languages'] = pd.Series(languagelist)
        dflangs = display_grid(dflang)

        langs = dflangs['languages'].to_list()
       
    if st.button('Generate plot:'):
        if nlang == 'One':
            whole_network_plotter(lang,iol,nwords,noc)
            
        else:
            for lang in langs:
                whole_network_plotter(lang,iol,nwords,noc)
                



if __name__ == "__main__":
    run_app()