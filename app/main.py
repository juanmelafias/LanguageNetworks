import streamlit as st
import os
import pandas as pd
import plotly.express as px

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode


def display_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: Pandas dataframe to be converted to AgGrid

    Returns:
        data: Modified pandas dataframe

    This module allows the app user to edit a dataframe in real time and save those
    changes in memory using the AgGrid module.
    """
    # Construct GridOptionsBuilder dict
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=True, groupable=True)
    #sel_mode = st.radio("Selection type", options=["single", "multiple"])
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd.build()
    
    Table = AgGrid(
        df,
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=500,
        allow_unsafe_jscode=True,
        theme="fresh",
    )
    # here only selected rows are used to generate reports. The problem is that
    # the returned value is a list of dictionaries, not a df, so we need to transform it back again
    data = pd.DataFrame(Table["selected_rows"])
    

    """
    Table = AgGrid(
        df,
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=500,
        allow_unsafe_jscode=True,
        theme="fresh",
    )
    data = Table["data"]
    """
    return data

def read_plot_info(language,nwords,iol):
    
    df = pd.read_csv(f'files/{iol}/dfplot/{language}.csv')
    df = df.drop(labels = ['Unnamed: 0'],axis = 1).sort_values(by='ranking')
    if nwords != 0:
        df = df.iloc[0:nwords]
    return df

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
        options = ['Yes','No'])
    if filteryes=='Yes':
        filter = st.selectbox('Filter by',
            (col for col in cols))
        filtervalue = st.selectbox(f'Select value of {filter} to Filter by',
            (col for col in df.groupby(by=filter).count().index))
        df = df[df[filter] == filtervalue]
    if st.button('Generate plot:'):

        #col = st.color_picker('Select a plot colour')
        
        fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3',
                                color=color, symbol=symbol, size = size, text = text , hover_name = extra)
        #fig.update_traces(marker=dict(color = col))

        st.plotly_chart(fig)



if __name__ == "__main__":
    run_app()
