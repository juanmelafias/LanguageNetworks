import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from utilsstreamlit import display_grid, read_plot_info


def run_app():

    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]

    st.title('Syntax networks plotter')

    iol = st.radio('Would you like to show data of inflected or lemmatized forms:',
        options = ['inflected','lemmatized'])
    pon = st.radio('Would you like to show data of primary or neighbour properties:',
        options = ['primaries','neighbours'])
    dim = st.radio('Would you like to show data on 2D or 3D:',
    options = ['2D','3D'])
        
    
    

    df = pd.read_csv(f'files/{iol}/languagesmean{pon}/dflangcomp.csv')
    df = df.drop(labels = ['Unnamed: 0'],axis = 1)
        
    

    
    cols = [col for col in df.columns]

    colors = {'r':'1','k':'2','b':'3','g':'4','y':'5'}
    if st.button('Generate plot:'):
        df['size'] = 30
        df['nc5'] = df['nc5'].apply(lambda x: colors[x])
        #col = st.color_picker('Select a plot colour')
        if dim == '3D':
            fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3',
                                    color='nc5', text = 'languages')
        else: 
            fig = px.scatter(df, x='pc1', y='pc2',
                                    color='nc5', hover_data = ['languages'],size = 'size')

        #fig.update_traces(marker=dict(color = col))

        st.plotly_chart(fig)



if __name__ == "__main__":
    run_app()
