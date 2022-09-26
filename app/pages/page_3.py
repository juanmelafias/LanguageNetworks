import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from common.utilsstreamlit import display_grid, read_plot_info


def run_app():

    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]

    st.title('Languages PCA plotter')

    iol = st.radio('Would you like to show data of inflected or lemmatized forms:',
        options = ['inflected','lemmatized'])
    pon = st.radio('Would you like to show data of primary or neighbour properties:',
        options = ['primaries','neighbours'])
    dim = st.radio('Would you like to show data on 2D or 3D:',
    options = ['2D','3D'])
        
    
    

    df = pd.read_csv(f'files/{iol}/languagesmean{pon}/dflangcomp.csv')
    df = df.drop(labels = ['Unnamed: 0'],axis = 1)
        

    compari = 'No'

    
    cols = [col for col in df.columns]

    colors = {'r':'1','k':'2','b':'3','g':'4','y':'5'}
    if compari == 'No':
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
    else:
        if st.button('Generate plot:'):
            
            if iol == 'inflected':
                dfin = df
                dfin['pc1_i'] = dfin['pc1']
                dfin['pc2_i'] = dfin['pc2']
                dfin['pc3_i'] = dfin['pc3']
                dflem = pd.read_csv(f'files/lemmatized/languagesmean{pon}/dflangcomp.csv')
                dflem = dflem.drop(labels = ['Unnamed: 0'],axis = 1)

                dflem['iol'] = 'lemmatized'
                dfin['iol'] = 'inflected'
                df = pd.concat([dfin,dflem],axis = 0,join = 'outer',ignore_index = True)
            else:
                dflem = df
                
                dfin = pd.read_csv(f'files/inflected/languagesmean{pon}/dflangcomp.csv')
                dfin = dfin.drop(labels = ['Unnamed: 0'],axis = 1)
                dfin['pc1_i'] = dfin['pc1']
                dfin['pc2_i'] = dfin['pc2']
                dfin['pc3_i'] = dfin['pc3']

                dflem['iol'] = 'lemmatized'
                dfin['iol'] = 'inflected'
                
                
                
                df = pd.concat([dfin,dflem],axis = 0,join = 'outer',ignore_index = True)

                
            df['size'] = 10
            df['nc5'] = df['nc5'].apply(lambda x: colors[x])
            #col = st.color_picker('Select a plot colour')
            print(df['iol'])
            if compari == 'No':
                if dim == '3D':
                    fig = px.scatter_3d(df, x='pc1_i', y='pc2_i', z='pc3_i',
                                            color='nc5', text = 'languages')
                else: 
                    fig = px.scatter(df, x='pc1_i', y='pc2_i',
                                        color='nc5', hover_data = ['languages'],size = 'size')
            else:
                if dim == '3D':
                    fig = px.scatter_3d(df, x='pc1_i', y='pc2_i', z='pc3_i',
                                            color='nc5', symbol = 'iol', text = 'languages')
                else: 
                    fig = px.scatter(df, x='pc1_i', y='pc2_i',
                                        color='nc5', symbol = 'iol', hover_data = ['languages'],size = 'size')


            #fig.update_traces(marker=dict(color = col))

            st.plotly_chart(fig)




if __name__ == "__main__":
    run_app()
