import streamlit as st
import os
import pandas as pd
import plotly.express as px


from common.constants import column_mapping, pos_mapping
from common.utilsstreamlit import read_plot_info
from common.tracking import log_main
from common.visitor_tracker import VisitorTracker

from loguru import logger

column_mapping_inv = {v: k for k, v in column_mapping.items()}
inverse_pos_mapping = {v: k for k, v in pos_mapping.items()}

def run_app():

    try:
        tracker = VisitorTracker()
        log_main(tracker, page_name="PCA Plotter")
    except Exception as e:
        logger.error(f"Error initializing visitor tracker: {e}")

    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]

    st.title('Words PCA plotter')

    
    # dflangs = display_grid(dflang)
    st.session_state.langs = st.multiselect('Pick languages to include:',
        options = (lang for lang in languagelist)
    )
    pc1 = 'PC1 Inflected Spanish'
    pc2 = 'PC2 Inflected Spanish'
    pc3 = 'PC3 Inflected Spanish'

    iol = st.radio('Would you like to show data of inflected or lemmatized forms:',
        options = ['inflected','lemmatized'],
        #Explain difference between inflected and lemmatized forms
        help= "Inflected forms are the different grammatical forms of a word (is, are), while lemmatized forms are the base or dictionary form of a word (be).")
    st.caption("💡 Words with the same part of speech share syntactic properties. This allows for fair comparisons across languages")
    st.session_state.pos = st.multiselect('Choose parts of speech to visualize:',
        options = (pos for pos in pos_mapping.values())
    )
    st.session_state.pos_mapped = [inverse_pos_mapping[pos] for pos in st.session_state.pos]
    # allorfew = st.radio('Would you like to create the network with all words or just a few?:',
    #     options = ['All','Custom'],
    #     help="Networks are created with the top 500 most common words. If you'd like a smaller network choose custom")
    # if allorfew == 'All':
    #     nwords = 0
    # else:
    #     nwords = st.text_input('Top N words to display')
    #     try:
    #         nwords = int(nwords)
    #     except ValueError:
    #         nwords = 500
    nwords = 0

 

        # langs = dflangs['languages'].to_list()

    color = 'Language'
    symbol = 'Part of Speech'
    size = 'Frequency'
    text = 'Translation'
    extra = 'Word'
    # filteryes = st.radio('Would you like to filter?:',
    #     options = ['No','Yes'])
    # if filteryes=='Yes':
    #     filter = st.selectbox('Filter by',
    #         (col for col in cols))
    #     filtervalue = st.selectbox(f'Select value of {filter} to Filter by',
    #         (col for col in df.groupby(by=filter).count().index))
    #     df = df[df[filter] == filtervalue]
    dim = st.radio('Would you like to show data in 2D or 3D:',
        options = ['2D','3D'])
    if st.session_state.langs:
        if st.button('Generate plot:'):

            df = pd.DataFrame()

            
            for lang in st.session_state.langs:
                df2concat = read_plot_info(lang,nwords,iol)
                df = pd.concat([df,df2concat],axis = 0,join = 'outer',ignore_index = True)
            
            df['ranking_inv'] = df['ranking'].apply(lambda x: abs(501-x))
            df['nc5'] = df['nc5'].apply(lambda x: str(x))
            df.rename(columns = column_mapping_inv, inplace = True)
            df = df[df['Part of Speech'].isin(st.session_state.pos_mapped)]
            
            #col = st.color_picker('Select a plot colour')
            if dim == "3D":
            
                fig = px.scatter_3d(df, x=pc1, y=pc2, z=pc3,
                                        color=color, symbol=symbol, size = size, text = text , hover_name = extra)
                #fig.update_traces(marker=dict(color = col))
            else:
                fig = px.scatter(df, x=pc1, y=pc2,
                                        color=color,  size = size, text = text , hover_name = extra)

            fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')

            st.caption("💡 Each bubble shows the properties of each node/word in the syntax network of its language projected in the Principal Component Eigenspace of inflected Spanish. Bubbles are coloured by Language, sized by Frequency, and show the original word translation when hovered over. You can zoom in and pan around the plot using your mouse.")

            st.plotly_chart(fig)



if __name__ == "__main__":
    run_app()
