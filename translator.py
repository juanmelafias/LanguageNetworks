import pandas as pd 
import googletrans 
import os
from utilsstreamlit import read_plot_info
translator = googletrans.Translator()
dictlang = googletrans.LANGCODES
dictlangcodes = {key.capitalize():value for key,value in dictlang.items()}
list_trans = [lang for lang in dictlangcodes.keys()]
filelist = os.listdir('./files/inflected/dictionaries/')
languagelist = [file.split('.')[0] for file in filelist]
languagelist = [lang for lang in languagelist if lang in list_trans]
for lang in languagelist[21:]:
    print(lang)
    for iol in ['inflected','lemmatized']:
        code = dictlangcodes[lang]
        df=read_plot_info(lang,0,iol)
        df['trans'] = df['palabra'].apply(lambda x:translator.translate(x,dest='en',src = code).text)
        df.to_csv(f'files/{iol}/dfplot/{lang}.csv')