from utils import stats_language
lemmatized = True
if lemmatized:
    direc = 'files/lemmatized/'
else:
    direc = 'files/inflected/'
stats = stats_language(lemmatized)
stats.to_csv(f'{direc}stats.csv')