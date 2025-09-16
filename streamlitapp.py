# use tabbar.py example instead of tabs

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json

import plotly.io as pio
pio.templates.default = 'seaborn' # "plotly"

# See https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart for selection examples
# using st.plotly_chart

st.set_page_config(layout='wide')
st.title('Exploring Colorado Moss List')


@st.cache_data
def load_no_dups(filename):
    with open(filename, 'rb') as f:
        merges, groups, df = pickle.load(f)
    counties = np.unique(merges['county']).tolist()
    counties.remove('')
    # species = np.unique(merges['acceptedName Bryonames']).tolist()
    species = [s.split() for s in np.unique(merges['acceptedName Bryonames'])]
    species = [' '.join(s[:2]) for s in species if s[1][0].islower()]
    return df, groups, merges, counties, species

@st.cache_data
def load_species_each_county(filename):
    with open(filename, 'rb') as f:
        species_each_county = pickle.load(f)
    return species_each_county

@st.cache_data
def load_counties_each_species(filename):
    with open(filename, 'rb') as f:
        counties_each_species = pickle.load(f)
    return counties_each_species

def gps_to_float(row, index):
    # index = 0 for lat, 1 for lon
    both = row.get('gps')
    try:
        lat_or_lon = eval(both)[index]
    except:
        lat_or_lon = 0.0
    return lat_or_lon

@st.cache_data
def load_deduplicated(filename):
    deduplicated = pd.read_csv(filename)
    # Remove authority from each accepted Name
    deduplicated['species'] = deduplicated['acceptedName'].apply(lambda gsa: ' '.join(gsa.split()[:2]))
    # convert gps to lat and lon
    deduplicated.loc[:, 'lat'] = deduplicated.apply(lambda row: gps_to_float(row, 0), axis=1)
    deduplicated.loc[:, 'lon'] = deduplicated.apply(lambda row: gps_to_float(row, 1), axis=1)
    # text to show when hovering
    eds = [(d if not pd.isna(d) else '') for d in deduplicated['eventDate']]
    deduplicated.loc[:, 'hovertext'] = [s + ' ' + ed for (s, ed) in zip(deduplicated['species'], eds)]  # 'eventDate']].agg(' '.join, axis=1)
    # add delta gps values to ones using county centers
    using_county_centers = deduplicated['countyCenterUsed'] == 1
    n = sum(using_county_centers)
    deduplicated.loc[using_county_centers,'lat'] += np.random.uniform(-0.01, 0.01, n)
    deduplicated.loc[using_county_centers,'lon'] += np.random.uniform(-0.01, 0.01, n)
    return deduplicated

# https://stackoverflow.com/questions/69396009/add-us-county-boundaries-to-a-plotly-density-mapbox
def get_county_boundaries():
    import urllib
    from pathlib import Path
    from zipfile import ZipFile
    import geopandas as gpd
    import requests

    # get geometry data as a geopandas dataframe
    src = [
        {
            "name": "counties",
            "suffix": ".shp",
            "url": "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_5m.zip",
        },
    ]
    data = {}
    for s in src:
        f = Path.cwd().joinpath(urllib.parse.urlparse(s["url"]).path.split("/")[-1])
        if not f.exists():
            r = requests.get(s["url"],stream=True,)
            with open(f, "wb") as fd:
                for chunk in r.iter_content(chunk_size=128): fd.write(chunk)

        fz = ZipFile(f)
        fz.extractall(f.parent.joinpath(f.stem))

        data[s["name"]] = gpd.read_file(
            f.parent.joinpath(f.stem).joinpath([f.filename
                                                for f in fz.infolist()
                                                if Path(f.filename).suffix == s["suffix"]][0])
        ).assign(source_name=s["name"])
    gdf = pd.concat(data.values()).to_crs("EPSG:4326")
    return gdf

######################################################################

df, groups, merges, counties, species = load_no_dups('no_dups.pkl') #df_groups.pkl')

# map_tab, counts_tab = st.tabs(('Map', 'Counts by Species\nand Counties'))
map_tab, samples_each_species_tab, species_each_county_tab, counties_each_species_tab, timeline_tab = st.tabs(
    ('Map', 'Samples Each Species', 'Species Each County', 'Counties each Species', 'Timeline'))

with map_tab:

    # with st.container():

        # col1, col2 = st.columns([1, 3])
        # with col1:

    use_counties = st.multiselect('Counties', counties)
    use_species = st.multiselect('Species', species)

    # use_counties = ['Larimer']
    # use_species = []

    deduplicated = load_deduplicated('deduplicated.csv')
    # selected = deduplicated.copy()
    selected = deduplicated

    if len(use_counties) > 0:
        selected = selected[selected['county'].isin(use_counties)]

    if len(use_species) > 0:
        selected = selected[selected['species'].isin(use_species)]

    #gps = selected[['gps', 'county', 'countyCenterUsed', 'acceptedName', 'species', 'date']]
    selected_with_gps = selected[~pd.isna(selected['gps'])]
    # ones_not_county_centers = ~pd.isna(selected_with_gps['countyCenterUsed'])
    # not_county_centers = selected_with_gps[ones_not_county_centers]  # pd.isna(selected_with_gps['countyCenterUsed'])]
    # with_county_centers = selected_with_gps[~ones_not_county_centers]  # selected_with_gps['countyCenterUsed'] == 1]

    fig = go.Figure(go.Scattermap(
        lat=selected_with_gps['lat'],  # must show all so indices selected work to show table
        lon=selected_with_gps['lon'],
        mode='markers',
        marker={'color': 'Green',
                'symbol': 'circle',
                'size': 10},
        text=selected_with_gps['hovertext'],
        hoverinfo='text',
        showlegend=False))

    if False:
        gdf = get_county_boundaries()
        with open('gdf.pkl', 'wb') as f:
            pickle.dump(gdf, f)
    else:
        with open('gdf.pkl', 'rb') as f:
            gdf = pickle.load(f)
        gdf = gdf[gdf['STATEFP']=='08']

    # get map from https://www.arcgis.com/apps/mapviewer/index.html?featurecollection=https%3A%2F%2Fbasemap.nationalmap.gov%2Farcgis%2Frest%2Fservices%3Ff%3Djson%26option%3Dfootprints&supportsProjection=true&supportsJSONP=true
        
    fig.update_layout(
        map_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "United States Geological Survey",
                "source": [
                    "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                ],
                'opacity': 0.3
            },
            {
                "source": json.loads(gdf.geometry.to_json()),
                "below": "traces",
                "type": "line",
                "color": "black",  #"yellow" if above map fro US Geological Survey is included
                "line": {"width": 0.5},
            },
        ],
        # mapbox_style="white-bg",
        hovermode='closest', 
        showlegend=False,
        height=800,
        # width=500,
        # map_layers=[
        #     {"below": 'traces',
        #      "sourcetype": "raster",
        #      "sourceattribution": "United States Geological Survey",
        #      "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"]
        #      }],

        map=dict(bearing=0,
                 center=go.layout.map.Center(
                     # lat=selected_with_gps['lat'].mean(),
                     # lon=selected_with_gps['lon'].mean()),
                     lat=39,
                     lon=-105.545167),
                 style="open-street-map",  # You can choose other styles like "open-street-map", "streets", "dark", etc.
                 pitch=0,
                 zoom=6),
    )

    event = st.plotly_chart(fig, on_select='rerun', key='selected_points', selection_mode=('points', 'box', 'lasso'),
                            use_container_width=False)

    picked_indices = event['selection']['point_indices']

    if len(picked_indices) == 0:
        # show all samples based on county and species
        selected_these = selected_with_gps

    else:
        # picked_indices
        # f'Number of selected ponts {len(picked_indices)}'
        # 'selected_with_gps.iloc[picked_indices]'

        selected_these = selected_with_gps.iloc[picked_indices]
        # f'{selected_these.shape=}'

    # selected_these = selected_with_gps.loc[picked_indices]
    show_columns = st.multiselect('Columns', selected_these.columns,
                                  default=['acceptedName', 'eventDate', 'county', 'recordedBy'])

    # selected_these.columns
    
    search_result_df = pd.DataFrame(selected_these[show_columns])
    # f'{search_result_df.shape=}'


    f'Selected {search_result_df.shape[0]:,d} samples.  Scroll down in the table to see all samples.'
    
    st.dataframe(search_result_df,
                 column_config={'packet': st.column_config.LinkColumn('packet',
                                                                      display_text='packet image')})

    

######################################################################

@st.cache_data
def make_species_counts_fig(percent_range=(75, 100)):
    species = deduplicated['acceptedName']
    species = [' '.join(s.split()[:2]) + ' ' for s in species]

    species = np.array(species)
    uniqs = np.unique(species)
    result = pd.DataFrame([[u, sum(species == u)] for u in uniqs], columns=('species', 'count'))
    result = result.sort_values('count', ascending=False)
    total_number = result['count'].sum()

    # fig = go.Histogram(y=species)
    # fig = px.histogram(result, y='species', x='count', orientation='h')
    high = float(len(result)) + 1
    low = high - high * 0.05
    fig = px.bar(result, y='species', x='count', orientation='h',
                 title=f'Number of samples for each species. Totals of {len(result)} species and {total_number:,} samples.',
                 height=1000,
                 range_y = [low, high],
                 barmode = 'overlay')
    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': None,
                             'automargin': True, 'dtick':1},
                      xaxis={'fixedrange': True},
                      # margin=dict(l=250),
                      dragmode='pan')

    return fig

@st.cache_data
def make_species_each_county_fig():
    species_each_county = load_species_each_county('species_each_county.pkl')
    result = []
    for co, sp in species_each_county.items():
        if co != '?':
            result.append([len(sp), co])

    # Add county names without any species
    counties = pd.read_csv('county-names.csv', dtype=str).values.astype(str)
    counties = [c[0] for c in counties]
    counties_in_results = [r[1] for r in result]
    for c in counties:
        if c not in counties_in_results:
            result.append([0, c])

    result = pd.DataFrame(result, columns=('Number of Species', 'County'))
    result = result.sort_values(by='Number of Species', ascending=True)

    high = float(len(result)) + 1
    low = high - high * 0.5
    xtext = f'Number of Species (Total of {len(species)} species)'
    fig = px.bar(result, x='Number of Species', y='County',
                 labels={'Number_of_Species': xtext},
                 title=f'Number of Species (out of {len(species)}) in Each County',
                 orientation='h', height=1000,
                 barmode = 'overlay',
                 range_y = [low, high])
    fig.update_layout(yaxis={'automargin': True, 'dtick':1, 'title': None},
                      xaxis={'fixedrange': True},
                      dragmode='pan')


    return fig

@st.cache_data
def make_county_counts_fig():
    counties_each_species = load_counties_each_species('counties_each_species.pkl')
    result = []
    for sp, counties_local in counties_each_species.items():
        result.append([len(counties_local), sp])
    # 1/0
    result = pd.DataFrame(result, columns=('Number of Counties', 'Species'))

    def totest():
        def gs(st):
            return ' '.join(st.split()[:2])

        for i in range (len(result) - 1):
            if gs(result.iloc[i]['Species']) == gs(result.iloc[i+1]['Species']):
                print(result['Species'].iloc[i:i+2].values.tolist())

        # z = np.array(result['Species'])
        # gs = list(map(lambda n: ' '.join(n.split()[:2]), z))
        # zu = np.unique(gs)
        # c = [(zi, np.sum(z == zi)) for zi in zu] 
        # [(zui, ci) for (zui, ci) in zip (zu, c)]
        # [z[np.where(zi == ci)] for (zi, ci) in zip (z, c)]
        

    result['Species'] = result['Species'].apply(lambda n: ' '.join(n.split()[:2]))
    result = result.sort_values(by=f'Number of Counties', ascending=True)
    # result = result[-100:]
    xtext = f'Number of Counties)'
    high = float(len(result)) + 1
    low = high - high * 0.05
    fig = px.bar(result, x='Number of Counties', y='Species',
                 labels={'Number of Counties': xtext},
                 title=f'Number of Counties With Each Species)',
                 orientation='h', height=1000,
                 range_y = [low, high],
                 barmode = 'overlay')
    fig.update_layout(yaxis={'automargin': True, 'dtick': 1, 'title': None},
                      xaxis={'fixedrange': True},
                      dragmode='pan')


    return fig

import datetime as dt
@st.cache_data
def make_timeline_fig():

    def to_datetime(date_str):
        return dt.datetime.strptime(date_str, '%Y-%m-%d')

    def all_to_datetime(df):
        dates = df.eventDate
        datetimes = []
        for d in dates:
            if not pd.isna(d):
                try:
                    junk = to_datetime(d)
                    # datetimes.append(to_datetime(d))
                    if d > '1805':
                        # some dates are 0003-01-01, 1800-01-01. 
                        datetimes.append(d)
                except:
                    pass
        return np.array(datetimes)

    datetimes = all_to_datetime(deduplicated)

    def count_by_month_and_by_year(datetimes, first_y, last_y):
        by_month = []
        by_year = []
        years = [d[:4] for d in datetimes]
        months = [d[5:7] for d in datetimes]
        for y in range(first_y, last_y + 1):
            sy = str(y)
            matches = [dy for dy  in years if dy == sy]
            count = len(matches)
            by_year.append([sy, count])

            for m in range(1, 13):
                sm = f'{m:02d}'
                matches = [(dy, dm) for dy, dm
                           in zip(years, months) if dy == sy and dm == sm]
                count = len(matches)
                by_month.append([f'{sy}-{sm:>02s}-01', count])

        return (pd.DataFrame(by_month, columns=('Date', 'count')),
                pd.DataFrame(by_year, columns=('Date', 'count')))

    by_month, by_year = count_by_month_and_by_year(datetimes, 1850, 2023)
    # by_month, by_year = count_by_month_and_by_year(datetimes, 1850, 2023)

    # px.line(x=counts['Date'], y=counts['count'])
    # px.bar(x=by_month['Date'], y=by_month['count'])

    fig = px.bar(x=by_year['Date'], y=by_year['count'],
                 labels={'x': 'Year', 'y': 'Number of Collections'})

    fig.update_xaxes(tickangle=45)
    return fig

# map_tab, samples_each_species_tab, species_each_county_tab, counties_each_species_tab

with samples_each_species_tab:    
    st.markdown(':green[Click and drag on chart to see more species.]')
    fig1 = make_species_counts_fig()
    st.plotly_chart(fig1)

    
with species_each_county_tab:
    st.markdown(':green[Click and drag on chart to see more counties.]')
    fig2 = make_species_each_county_fig()
    st.plotly_chart(fig2)


with counties_each_species_tab:
    st.markdown(':green[Click and drag on chart to see more species.]')
    fig3 = make_county_counts_fig()
    st.plotly_chart(fig3)

with timeline_tab:
    st.markdown(':green[Click and drag on chart to zoom in]')

    fig4 = make_timeline_fig()
    st.plotly_chart(fig4)
