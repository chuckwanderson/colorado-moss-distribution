###  dups_scientific..... make list of all packets and institution codes for collectiosn with duplicates

import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json
import math
import geopandas

import plotly.io as pio
pio.templates.default = 'seaborn' # "plotly"

# See https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart for selection examples
# using st.plotly_chart

st.set_page_config(layout='wide')
st.title('Exploring Colorado Bryophytes')
st.warning(
    body='*:red[Warning: This web page and associated data are under development.]*',
    icon=":material/warning:"
)

# @st.cache_data
# def load_no_dups(filename):
#     with open(filename, 'rb') as f:
#         merges, groups, df = pickle.load(f)
#     counties = np.unique(merges['county']).tolist()
#     counties.remove('')
#     # species = np.unique(merges['acceptedName Bryonames']).tolist()
#     species = [s.split() for s in np.unique(merges['acceptedName Bryonames'])]
#     species = [' '.join(s[:2]) for s in species if s[1][0].islower()]
#     return df, groups, merges, counties, species

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
def make_species_counts():
    species = deduplicated['acceptedNameUsage']
    # species = [' '.join(s.split()[:2]) + ' ' for s in species]

    species = np.array(species)
    uniqs = np.unique(species)
    result = pd.DataFrame([[u, sum(species == u)] for u in uniqs], columns=('species', 'count'))
    result = result.sort_values('count', ascending=False)
    return result

@st.cache_data
def load_deduplicated(filename):
    deduplicated = pd.read_csv(filename, sep='\t')
    # Remove authority from each accepted Name
    deduplicated['species'] = deduplicated['acceptedNameUsage'].apply(lambda gsa: ' '.join(gsa.split()[:2]))
    # convert gps to lat and lon
    gpses = {'packet': deduplicated['gps_packet'],
             'county': deduplicated['gps_county'],
             'verbatim': deduplicated['gps_verbatim'],
             'llm': deduplicated.apply(lambda row: (row['latitude_llm'], row['longitude_llm']), axis=1)}
    df_gpses = pd.DataFrame(gpses)
    def pick_gps(row):
        if eval(row['packet']) != (0, 0):
            # print('packet', eval(row['packet']))
            return [eval(row['packet']), 'packet']
        if eval(row['verbatim']) != (0, 0):
            return [eval(row['verbatim']), 'verbatim']
        if not (math.isnan(row['llm'][0]) and math.isnan(row['llm'][1])):
            # print('llm', row['llm'])
            if row['llm'][0] < -10:
                row['llm'] = (row['llm'][0] * -1, row['llm'][1])
            return [row['llm'], 'llm']
        # print('county', eval(row['county']))
        return [eval(row['county']), 'county']

    deduplicated[['use_this_gps', 'gps_source']] = df_gpses.apply(lambda row: pick_gps(row), axis=1, result_type='expand')
    nope = deduplicated[deduplicated['use_this_gps'].apply(lambda gps: not isinstance(gps[0], float))]
    good_gps_count = sum(deduplicated['use_this_gps'].apply(lambda gps: isinstance(gps[0], float)))
    print(f'{good_gps_count} good gps values out of {deduplicated.shape[0]}')

    # deduplicated.loc[:, 'lat'] = deduplicated.apply(lambda row: gps_to_float(row, 0), axis=1)
    # deduplicated.loc[:, 'lon'] = deduplicated.apply(lambda row: gps_to_float(row, 1), axis=1)
    # text to show when hovering
    # eds = [(d if not pd.isna(d) else '') for d in deduplicated['eventDate']]
    # deduplicated.loc[:, 'hovertext'] = [s + ' ' + ed for (s, ed) in zip(deduplicated['species'], eds)]  # 'eventDate']].agg(' '.join, axis=1)
    deduplicated.loc[:, 'hovertext'] = deduplicated['species']
    # add delta gps values to ones using county centers
    # using_county_centers = deduplicated['countyCenterUsed'] == 1
    # n = sum(using_county_centers)
    # deduplicated.loc[using_county_centers,'lat'] += np.random.uniform(-0.01, 0.01, n)
    # deduplicated.loc[using_county_centers,'lon'] += np.random.uniform(-0.01, 0.01, n)
    return deduplicated


# https://stackoverflow.com/questions/69396009/add-us-county-boundaries-to-a-plotly-density-mapbox
def get_census_gov_county_boundaries(state_name):
    import urllib
    from pathlib import Path
    from zipfile import ZipFile
    import geopandas as gpd
    import requests

    #  from list of FIPS codes at https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html#accordion-455dd4b1cd-item-2c5946f1b7
    fips_codes = {
        'Colorado': '08',
        'New Mexico': '35',
        'Arizona': '04'
        }

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

    if state_name not in fips_codes:
        print(f'State {state_name} notin fips_codes {fips_codes}')
    else:
        gdf = gdf[gdf['STATEFP'] == fips_codes['New Mexico']]
        with open('gdf.pkl', 'wb') as f:
            pickle.dump(gdf, f)

@st.cache_data
def get_county_boundaries():
    state_name = 'Colorado'
    get_census_gov_county_boundaries(state_name)
    with open('gdf.pkl', 'rb') as f:
        gdf = pickle.load(f)
    return gdf[gdf['STATEFP'] == '08']


# SETUP #####################################################################
deduplicated = load_deduplicated('deduplicated.tsv')
species_each_county = load_species_each_county('species_each_county.pkl')
counties_each_species = load_counties_each_species('counties_each_species.pkl')
species_counts = make_species_counts()
######################################################################

# Sidebar #####################################################################

if "sidebar_version" not in st.session_state:
    st.session_state['sidebar_version'] = 0

def reset_sidebar_selections():
    st.session_state['sidebar_version'] += 1

sidebar_version = st.session_state['sidebar_version']

# def clear_session_state():
#     st.session_state.selected_counties = []
#     st.session_state.slider_version += 1
#     st.session_state.selected_species = []
#     st.session_state.selected_recordedBy = ''
    

#     for key in st.session_state.keys():
#         del st.session_state[key]
        
with st.sidebar:

    st.title('Map and Table Selections')

    # Change nan dates to '?'.  Needed for creating range of event dates.
    nan_dates = [(i, d) for d in enumerate(deduplicated['eventDateFixed']) if isinstance(d, float)]
    deduplicated.loc[nan_dates, 'eventDateFixed'] = '?'

    # Remove entries without GPS coordinates
    selected = deduplicated
    selected_with_gps = selected[(~pd.isna(selected['use_this_gps'])) & (selected['use_this_gps'] != (0, 0)) & (selected['use_this_gps'] != (-1, -1))]

    ## Counties
    counties = list(deduplicated['county'].unique())
    counties = [c for c in counties if isinstance(c, str)]  #  and not math.isnan(c)]
    use_counties = st.multiselect('Counties', counties, key=f'selected_counties_{sidebar_version}')
    if len(use_counties) > 0:
        selected_with_gps = selected_with_gps[selected_with_gps['county'].isin(use_counties)]   # st.session_state.selected_counties)]

    ## Species times collected
    use_species_n_times = None
    counts = species_counts['count']
    min_count, max_count = counts.min(), counts.max()
    use_species_n_times = st.slider(
        "Select a range of number of times species collected:",
        min_value=min_count,
        max_value=max_count,
        value=(min_count, max_count),  # Default initial range
        key=f'selected_times_collected_{sidebar_version}')

    sp_count = use_species_n_times  # st.session_state[f'selected_times_collected_{st.session_state.slider_version}']
    if sp_count:
        use_species = species_counts[(sp_count[0] <= species_counts['count']) &
                                     (species_counts['count'] <= sp_count[1])]['species']
        selected_with_gps = selected_with_gps[selected_with_gps['species'].isin(use_species)]

    ## Species
    species = selected_with_gps['species'].unique()
    use_species = st.multiselect('Species', species, key=f'selected_species_{sidebar_version}')
    # if len(st.session_state.selected_species) > 0:
    if len(use_species) > 0:  # st.session_state.selected_species) > 0:
        selected_with_gps = selected_with_gps[selected_with_gps['species'].isin(use_species)]  # st.session_state.selected_species)]

    ## Event date
    def fix_dates(dstr):
        # print(dstr, end=' -> ')
        # if dstr[:4] == '1873':
            # print(dstr)
        if dstr[4] != '-':
            dstr = dstr[:4] + '-' + dstr[5:]
        if len(dstr) == 8:  # 20070114
            dstr = dstr[:4] + '-' + dstr[4:6] + '-' + dstr[6:8]
        if dstr.startswith('0000'):
            dstr = '2000' + dstr[4:]
        if dstr.endswith('00-00'):
            dstr = dstr[:4] + ' 01-01'
        if dstr.endswith('00'):
            dstr = dstr[:8] + '01'
        if dstr[5:7] == '00':
            dstr = dstr[:4] + '01' + dstr[8:]
        # print(dstr)
        return dstr

    selected_with_gps = selected_with_gps[selected_with_gps['eventDateFixed'].notna()]
    selected_with_gps = selected_with_gps[selected_with_gps['eventDateFixed'].str.count('-') == 2]
    selected_with_gps['eventDateFixed'] = selected_with_gps['eventDateFixed'].apply(fix_dates)
    selected_with_gps['eventDateFixed'] = pd.to_datetime(selected_with_gps['eventDateFixed'], format='mixed')
    dates = selected_with_gps['eventDateFixed'].unique()
    dates = [d for d in dates.to_pydatetime() if not isinstance(d, float)]
    min_d = min(dates)
    max_d = max(dates)
    use_dates = st.slider(
        "Select a range of dates:",
        min_value = min_d,
        max_value = max_d,
        value=(min_d, max_d),
        key=f'selected_dates_{sidebar_version}')
    # use_dates = st.session_state.selected_dates
    if use_dates:
        # selected_with_gps_with_date = selected_with_gps[selected_with_gps['eventDateFixed'].notna()]
        selected_with_gps = selected_with_gps[(use_dates[0] <= selected_with_gps['eventDateFixed']) &
                                              (selected_with_gps['eventDateFixed'] <= use_dates[1])]

    ## recordedBy
    use_name = st.text_input('Enter a name of a collector.', '', key=f'selected_recordedBy_{sidebar_version}')
    recordedBys = selected_with_gps['recordedBy']
    if use_name != '':
        selected_with_gps = selected_with_gps[(pd.notna(recordedBys)) & (recordedBys.str.contains(use_name))]  # st.session_state.selected_recordedBy))]
    
    ## # duplicates
    use_n_dups_range = None
    n = deduplicated['# duplicates'].unique()
    min_n = min(n)
    max_n = max(n)
    use_n_dups = st.slider(
        "Select a range of number of duplicates:",
        min_value=min_n,
        max_value=max_n,
        value=(min_n, max_n),  # Default initial range
        key=f'selected_num_duplicates_{sidebar_version}')
    # use_n_dups = st.number_input("Choose number of duplicates:",
    #                              min_value=min(n), max_value=max(n),
    #                              value=10,  # Default starting value
    #                              step=1)  # Forces integer increments
    # n_dups = st.session_state.selected_num_duplicates
    if use_n_dups:
        selected_with_gps = selected_with_gps[(use_n_dups[0] <= selected_with_gps['# duplicates']) &
                                              (selected_with_gps['# duplicates'] <= use_n_dups[1])]

    
    st.button('Clear all selections', on_click=reset_sidebar_selections)
        
######################################################################
######################################################################

map_tab, samples_each_species_tab, species_each_county_tab, counties_each_species_tab, timeline_tab = st.tabs(
    ('Map and Table', 'Samples Each Species', 'Species Each County', 'Counties each Species', 'Timeline'))

with map_tab:

    # with st.container():

        # col1, col2 = st.columns([1, 3])
        # with col1:

    # ones_not_county_centers = ~pd.isna(selected_with_gps['countyCenterUsed'])
    # not_county_centers = selected_with_gps[ones_not_county_centers]  # pd.isna(selected_with_gps['countyCenterUsed'])]
    # with_county_centers = selected_with_gps[~ones_not_county_centers]  # selected_with_gps['countyCenterUsed'] == 1]

    gps_df = selected_with_gps['use_this_gps']
    print('type', type(gps_df))
    fig = go.Figure(go.Scattermap(
        lat=gps_df.apply(lambda x: x[0]),
        lon=gps_df.apply(lambda x: x[1]),
        mode='markers',
        marker={'color': 'Black',
                #'symbol': 'circle',
                #'borderwidth': 3,
                'size': 8},
        hoverinfo='text',
        showlegend=False))

    fig.add_trace(go.Scattermap(
        lat = gps_df.apply(lambda x: x[0]), # must show all so indices selected work to show table
        lon = gps_df.apply(lambda x: x[1]),
        mode='markers',
        marker={'color': 'Yellow', ##F58700',
                #'symbol': 'circle',
                #'borderwidth': 3,
                'size': 6},
        selected=dict(marker=dict(color='yellow', opacity=1.0, size=6)),
        unselected=dict(marker=dict(color='orange', opacity=0.05, size=6)),
        text=selected_with_gps['hovertext'],
        # hoverinfo='text',
        showlegend=False))


    # get map from https://www.arcgis.com/apps/mapviewer/index.html?featurecollection=https%3A%2F%2Fbasemap.nationalmap.gov%2Farcgis%2Frest%2Fservices%3Ff%3Djson%26option%3Dfootprints&supportsProjection=true&supportsJSONP=true
        
    gdf = get_county_boundaries()
    
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

    st.text('Use the box or lasso selection tools to select samples to appear in the table below the map. Double click on the map to clear your current selection.')

    event = st.plotly_chart(fig, on_select='rerun',
                            key='selected_points',
                            selection_mode=('points', 'box', 'lasso'),
                            use_container_width=False,
                            config={'displayModeBar': True},
                            )
                            # width=False)

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

    selected_these = selected_these.rename(columns={'references': 'packets'})

    # selected_these = selected_with_gps.loc[picked_indices]
    show_columns = st.multiselect('Columns', selected_these.columns,
                                  default=['acceptedNameUsage', 'eventDate', 'county', 'recordedBy', 'use_this_gps', 'gps_source'])

    # selected_these.columns
    
    search_result_df = pd.DataFrame(selected_these[show_columns])
    # f'{search_result_df.shape=}'


    f'Selected {search_result_df.shape[0]:,d} samples.  Scroll down in the table to see all samples. Click on a column heading to sort by that column.'

    st.dataframe(search_result_df, hide_index=True,
                 column_config={'packets': st.column_config.LinkColumn('links to packets',
                                                                       display_text='packet')})

    csv_results = search_result_df.to_csv(index=False).encode('utf-8')
    filename = 'moss_results.csv'

    # with st.container(border=False):
    #     # Inject CSS targeting buttons inside THIS container structure
    #     st.markdown(
    #         """
    #         <style>
    #         /* Target the download button inside the container */
    #         div[data-testid="stVerticalBlock"] div[data-testid="stElementContainer"] button[data-testid="stDownloadButton"] {
    #             background-color: #008800B; /* Change to your preferred hex color */
    #             color: #DD0000;              /* Text color */
    #             border-radius: 8px;        /* Optional: rounded corners */
    #             border: none;
    #         }

    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )

    st.download_button(label=f'**Download this table to {filename}**',
                       data=csv_results,
                       file_name=filename,
                       mime='text/csv')



######################################################################


@st.cache_data
def make_species_counts_fig(percent_range=(75, 100)):
    total_number = species_counts['count'].sum()

    # fig = go.Histogram(y=species)
    # fig = px.histogram(result, y='species', x='count', orientation='h')
    high = float(len(species_counts)) + 1
    low = high - high * 0.05
    fig = px.bar(species_counts, y='species', x='count', orientation='h',
                 title=f'Number of samples for each species. Totals of {len(species_counts)} species and {total_number:,} samples.',
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
    xtext = f'Number of Counties'
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

    # fig = px.bar(x=by_year['Date'], y=by_year['count'],
    #              labels={'x': 'Year', 'y': 'Number of Collections'})

    fig = px.bar(x=by_year['count'], y=by_year['Date'],
                 height=2000,
                 labels={'x': 'Number of Collections', 'y': 'Year'})

    fig.update_yaxes(type='category', dtick=1)
    
    fig.update_xaxes(tickangle=45)
    return fig

# map_tab, samples_each_species_tab, species_each_county_tab, counties_each_species_tab

with samples_each_species_tab:    
    st.markdown(':green[Click and drag on chart to see more species.]')
    fig1 = make_species_counts_fig()
    st.plotly_chart(fig1, config={'displayModeBar': True})

    
with species_each_county_tab:
    st.markdown(':green[Click and drag on chart to see more counties.]')
    fig2 = make_species_each_county_fig()
    st.plotly_chart(fig2, config={'displayModeBar': True})


with counties_each_species_tab:
    st.markdown(':green[Click and drag on chart to see more species.]')
    fig3 = make_county_counts_fig()
    st.plotly_chart(fig3, config={'displayModeBar': True})

with timeline_tab:
    st.markdown(':green[Click and drag on chart to zoom in]')

    fig4 = make_timeline_fig()
    st.plotly_chart(fig4, config={'displayModeBar': True})
