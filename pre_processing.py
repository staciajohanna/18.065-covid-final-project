import pandas as pd


end_date = '4/21/2020'
covid = pd.read_csv('deaths.csv')
covid['Deaths'] = covid.groupby(['Country/Region'])[end_date].transform('sum')
covid = covid.drop_duplicates(subset=['Country/Region'])
covid = covid.filter(items = ['Country/Region', 'Deaths'])

sars = pd.read_csv('sars.csv')
sars.drop_duplicates(subset = 'Country', keep = 'last', inplace = True)
sars = sars.filter(items = ['Country', 'Number of deaths'])

combined = covid.set_index('Country/Region').join(sars.set_index('Country'))
combined = combined.rename(columns={"Deaths": "COVID", "Number of deaths": "SARS"})
combined.to_csv('combined.csv')

# Conflicts that have to be manually addressed
conflicts = []
country_set = set([country for country in covid['Country/Region']])
for country, deaths in zip(sars['Country'], sars['Number of deaths']):
	if country not in country_set:
		conflicts.append((country, deaths))

# [('Hong Kong SAR, China', 298), ('Macao SAR, China', 0), ('Taiwan, China', 84), ('Republic of Ireland', 0), ('Republic of Korea', 0), ('Russian Federation', 0), ('United States', 0), ('Viet Nam', 5)]

