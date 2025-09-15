from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})

df['unemp'] = 100
import plotly.express as px

fig = px.choropleth(df, geojson=counties, locations='fips',
                     color='unemp',
                           # color_continuous_scale="Viridis",
                           range_color=(100, 100),
                           scope="usa",
                          # labels={'unemp':'unemployment rate'},                 
                          )
fig.update_traces(marker_line_width=1, marker_opacity=0.8)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_geos(
showsubunits=True, subunitcolor="black"
)
fig.show()
