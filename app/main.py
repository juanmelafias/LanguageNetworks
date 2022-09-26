import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from common.utilsstreamlit import display_grid, read_plot_info


def run_app():

    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]

    st.title('Words PCA plotter')

    iol = st.radio('Would you like to show data of inflected or lemmatized forms:',
        options = ['inflected','lemmatized'])
    allorfew = st.radio('Would you like to show all words or just a few?:',
        options = ['All','Custom'])
    if allorfew == 'All':
        nwords = 0
    else:
        nwords = st.text_input('Input desired number of words')
        nwords = int(nwords)
    nlang = st.radio('Would you like two show one or several languages?:',
        options = ['One','Several'])
    if nlang == 'One':
        lang = st.selectbox('Pick a language:',
            (lang for lang in languagelist))
        df = read_plot_info(lang,nwords,iol)
        pc1 = 'pc1'
        pc2 = 'pc2'
        pc3 = 'pc3'
        
    else:
        dflang = pd.DataFrame()
        dflang['languages'] = pd.Series(languagelist)
        dflangs = display_grid(dflang)
        pc1 = 'pc1is'
        pc2 = 'pc2is'
        pc3 = 'pc3is'

        langs = dflangs['languages'].to_list()
        df = pd.DataFrame()
        for lang in langs:
            df2concat = read_plot_info(lang,nwords,iol)
            df = pd.concat([df,df2concat],axis = 0,join = 'outer',ignore_index = True)

    df['ranking_inv'] = df['ranking'].apply(lambda x: abs(501-x))
    cols = [col for col in df.columns]

    color = st.selectbox('Pick a variable to represent color in the viz:',
            (col for col in cols))
    symbol = st.selectbox('Pick a variable to represent symbol in the viz:',
            (col for col in cols))
    size = st.selectbox('Pick a variable to represent size in the viz:',
            (col for col in cols))
    text = st.selectbox('Pick a variable to represent text in the viz:',
            (col for col in cols))
    extra = st.selectbox('Any other data to show while hovering',
            (col for col in cols))
    filteryes = st.radio('Would you like to filter?:',
        options = ['No','Yes'])
    if filteryes=='Yes':
        filter = st.selectbox('Filter by',
            (col for col in cols))
        filtervalue = st.selectbox(f'Select value of {filter} to Filter by',
            (col for col in df.groupby(by=filter).count().index))
        df = df[df[filter] == filtervalue]
    dim = st.radio('Would you like to show data in 2D or 3D:',
        options = ['2D','3D'])
    if st.button('Generate plot:'):
        
        df['nc5'] = df['nc5'].apply(lambda x: str(x))
        #col = st.color_picker('Select a plot colour')
        if dim == "3D":
        
            fig = px.scatter_3d(df, x=pc1, y=pc2, z=pc3,
                                    color=color, symbol=symbol, size = size, text = text , hover_name = extra)
            #fig.update_traces(marker=dict(color = col))
        else:
            fig = px.scatter(df, x=pc1, y=pc2,
                                    color=color,  size = size, text = text , hover_name = extra)

        fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')
        st.plotly_chart(fig)



if __name__ == "__main__":
    run_app()
