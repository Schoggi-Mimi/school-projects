# streamlit run Oliver/streamlit/webapp.py 
import data.data as data
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pycountry


def get_choropleth_life(df):
    fig = px.choropleth(df, locations="ISO", color="Lebenserwartung",
                        range_color=[df['Lebenserwartung'].min()-1, df['Lebenserwartung'].max()+1], animation_frame='Jahr',
                        hover_name="Land",
                        color_continuous_scale=px.colors.diverging.RdYlGn, width=1000, height=700)
    fig.update_traces(colorbar_nticks=10, selector=dict(type="choropleth"))
    fig.update_layout(geo=dict(bgcolor= 'rgba(0,0,0,0)'))
    return fig

def get_choropleth_water(df):
    fig = px.choropleth(df, locations="ISO", color="Trinkwasser Zugang",
                        range_color=[df['Trinkwasser Zugang'].min()-1, df['Trinkwasser Zugang'].max()+1], animation_frame='Jahr',
                        hover_name="Land",
                        color_continuous_scale=px.colors.diverging.RdYlGn, width=1000, height=700)
    fig.update_traces(colorbar_nticks=10, selector=dict(type="choropleth"))
    return fig

def get_choropleth_infant(df):
    fig = px.choropleth(df, locations="ISO", color="Säuglingssterberate",
                        range_color=[df['Säuglingssterberate'].min()-1, df['Säuglingssterberate'].max()+1], animation_frame='Jahr',
                        hover_name="Land",
                        color_continuous_scale=px.colors.diverging.RdYlGn[::-1], width=1000, height=700)
    fig.update_traces(colorbar_nticks=10, selector=dict(type="choropleth"))
    return fig
# ----------------------------------------------------------------
size_country = 20
size_continent = 50
size_npo = 30
opacity_country = 0.3
opacity_continent = 1
opacity_npo = 0.6
line_color = "white"
# ----------------------------------------------------------------
country_styles = {
    'Africa': {
        'marker': {'color': '#B279D7', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'Asia': {
        'marker': {'color': '#DE621E', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'Europe': {
        'marker': {'color': '#46ACA4', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'North America': {
        'marker': {'color': '#93F5E8', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'Oceania': {
        'marker': {'color': '#3673FE', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'South America': {
        'marker': {'color': '#AAEC71', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'Antarctica': {
        'marker': {'color': '#FCFCCF', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
    'Seven seas (open ocean)': {
        'marker': {'color': '#57A3B3', 'size': size_country, 'opacity': opacity_country, 'line': {'color': line_color, 'width': 1}},
    },
}

continent_styles = {
    'Africa': {
        'marker': {'color': '#B279D7', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'Asia': {
        'marker': {'color': '#DE621E', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'Europe': {
        'marker': {'color': '#46ACA4', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'North America': {
        'marker': {'color': '#93F5E8', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'Oceania': {
        'marker': {'color': '#3673FE', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'South America': {
        'marker': {'color': '#AAEC71', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'Antarctica': {
        'marker': {'color': '#FCFCCF', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
    'Seven seas (open ocean)': {
        'marker': {'color': '#57A3B3', 'size': size_continent, 'opacity': opacity_continent, 'symbol': 'octagon', 'line': {'color': line_color, 'width': 1}},
    },
}

npo_styles = {
    'splash': {
        'marker': {'color': '#15B0D1', 'size': size_npo, 'opacity': opacity_npo, 'line': {'color': line_color, 'width': 1}},
    },
    'Lifewater': {
        'marker': {'color': '#FD636C', 'size': size_npo, 'opacity': opacity_npo, 'line': {'color': line_color, 'width': 1}},
    },
    'water for good': {
        'marker': {'color': '#FFBA00', 'size': size_npo, 'opacity': opacity_npo, 'line': {'color': line_color, 'width': 1}},
    },
    'Blood:Water': {
        'marker': {'color': '#39E8B1', 'size': size_npo, 'opacity': opacity_npo, 'line': {'color': line_color, 'width': 1}},
    },
    'Pure Water for the World': {
        'marker': {'color': '#6967CF', 'size': size_npo, 'opacity': opacity_npo, 'line': {'color': line_color, 'width': 1}},
    },
}

def get_scatter_water_life_country(df_water_life, year=2015):
    iso3_to_iso2 = {c.alpha_3: c.alpha_2 for c in pycountry.countries}

    df = df_water_life.query("Jahr == @year")
    df["ISO_alpha2"] = df["ISO"].map(iso3_to_iso2)

    fig = px.scatter(
        df,
        x="Lebenserwartung",
        y="Trinkwasser Zugang",
        hover_name="Land",
        hover_data=["Lebenserwartung", "Trinkwasser Zugang", "Bevölkerung"],
    )
    fig.update_traces(marker_color="rgba(0,0,0,0)")

    minDim = df[["Lebenserwartung", "Trinkwasser Zugang"]].max().idxmax()
    maxi = df[minDim].max()
    for i, row in df.iterrows():
        country_iso = row["ISO_alpha2"]
        fig.add_layout_image(
            dict(
                source=f"https://raw.githubusercontent.com/matahombres/CSS-Country-Flags-Rounded/master/flags/{country_iso}.png",
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=row["Lebenserwartung"],
                y=row["Trinkwasser Zugang"],
                sizex=np.sqrt(row["Bevölkerung"] / df["Bevölkerung"].max()) * maxi * 0.15 + maxi * 0.03,
                sizey=np.sqrt(row["Bevölkerung"] / df["Bevölkerung"].max()) * maxi * 0.15+ maxi * 0.03,
                sizing="contain",
                opacity=0.8,
                layer="above"
            )
        )

    fig.update_layout(height=600, width=1000, plot_bgcolor="#dfdfdf", yaxis_range=[-5e3, 55e3])
    return fig

def get_scatter(df, y_axis, df_switzerland, year=2015):
    traces = []
    for ct in df['Kontinent'].unique():
        tmp_data = df.query('Kontinent == @ct')
        tmp_trace = {
            'x': tmp_data[tmp_data['Jahr'] == year][y_axis].to_list(), 
            'y': tmp_data[tmp_data['Jahr'] == year]['Trinkwasser Zugang'].to_list(),
            'type': 'scatter',
            'name': ct, 
            'mode': 'markers', 
            'text': tmp_data[tmp_data['Jahr'] == year]['Land'].tolist(),
            'showlegend': False,
            'hovertemplate': '<span style="font-weight:bold;">Country: %{text}</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
            'hoverlabel': {
                'bgcolor': 'black', 
                'bordercolor': country_styles[ct]['marker']['color'] # you should probably try-except this
            } 
        }
        tmp_trace.update(country_styles[ct]) # here we add additional keys to the dictionary
        traces.append(tmp_trace)

    trace_switzerland = {
        'name': 'Switzerland',
        'type': 'scatter',
        'mode': 'markers',
        'x': df_switzerland[df_switzerland['Jahr'] == year][y_axis].to_list(), 
        'y': df_switzerland[df_switzerland['Jahr'] == year]['Trinkwasser Zugang'].to_list(),
        'marker': {
            'size': 15,
            'symbol': 'square-cross',
            'line': {'width': 2, 'color': 'red'},
            'color': 'white',
        },
        'hovertemplate': '<span style="font-weight:bold;">Kontinent: Switzerland</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
        'opacity': 1,
        'legendgroup': "group",  # this can be any string, not just "group"
        'legendgrouptitle_text': "Vergleich Schweiz",
    }
    traces.append(trace_switzerland)

    water_life_continent = df[df['Jahr'] == year].groupby(['Kontinent'])[y_axis, 'Trinkwasser Zugang', 'Bevölkerung'].mean().reset_index()

    for ct in water_life_continent['Kontinent'].unique():
        tmp_data = water_life_continent.query('Kontinent == @ct')
        tmp_trace = {
            'x': tmp_data[y_axis].to_list(), 
            'y': tmp_data['Trinkwasser Zugang'].to_list(),
            'type': 'scatter',
            'name': ct, 
            'mode': 'markers', 
            'text': tmp_data['Kontinent'].tolist(),
            'hovertemplate': '<span style="font-weight:bold;">Kontinent: %{text}</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
            'hoverlabel': {
                'bgcolor': 'black', 
                'bordercolor': continent_styles[ct]['marker']['color'] # you should probably try-except this
            },
            'legendgroup': "group2",  # this can be any string, not just "group"
            'legendgrouptitle_text': "Kontinente",
        }
        tmp_trace.update(continent_styles[ct]) # here we add additional keys to the dictionary
        traces.append(tmp_trace)

    layout = {
        'title': f'{y_axis} versus Trinkwasser Zugang im Jahr {year}',
        'xaxis': {
            'range': [df[df['Jahr'] == year][y_axis].min()-1, df[df['Jahr'] == 2015][y_axis].max()+1],
            'showgrid': False, 
            'title': f'{y_axis} [%]'
            }, 
        'yaxis': {
            'linecolor': 'green', 
            'linewidth': 2, 
            'title': 'Trinkwasser Zugang [%]',
            'mirror': True
            },
        'height': 800,
        # 'width': 1700,
        'template': 'plotly_dark',
        'legend': {'groupclick': 'toggleitem'},
    }

    figdict = {'data': traces, 'layout': layout}
    figchart = go.Figure(**figdict)
    return figchart

def get_scatter_organization(df_organization, y_axis, df_switzerland, df, year=2015):
    traces = []
    for npo in df_organization['NPO'].unique():
        tmp_data = df_organization.query('NPO == @npo')
        tmp_trace = {
            'x': tmp_data[tmp_data['Jahr'] == year][y_axis].to_list(), 
            'y': tmp_data[tmp_data['Jahr'] == year]['Trinkwasser Zugang'].to_list(),
            'type': 'scatter',
            'name': npo, 
            'mode': 'markers', 
            'text': tmp_data[tmp_data['Jahr'] == year]['Land'].tolist(),
            'hovertemplate': '<span style="font-weight:bold;">Country: %{text}</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
            'hoverlabel': {
                'bgcolor': 'black', 
                'bordercolor': npo_styles[npo]['marker']['color'] # you should probably try-except this
            },
            'legendgroup': "group",  # this can be any string, not just "group"
            'legendgrouptitle_text': "Nonprofit organization",
        }
        tmp_trace.update(npo_styles[npo]) # here we add additional keys to the dictionary
        traces.append(tmp_trace)

    trace_switzerland = {
        'name': 'Switzerland',
        'type': 'scatter',
        'mode': 'markers',
        'x': df_switzerland[df_switzerland['Jahr'] == year][y_axis].to_list(), 
        'y': df_switzerland[df_switzerland['Jahr'] == year]['Trinkwasser Zugang'].to_list(),
        'marker': {
            'size': 15,
            'symbol': 'square-cross',
            'line': {'width': 2, 'color': 'red'},
            'color': 'white',
        },
        'hovertemplate': '<span style="font-weight:bold;">Land: Switzerland</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
        'opacity': 1,
        'legendgroup': "group2",  # this can be any string, not just "group"
        'legendgrouptitle_text': "Vergleich Schweiz",
    }
    traces.append(trace_switzerland)

    water_life_continent = df[df['Jahr'] == year].groupby(['Kontinent'])[y_axis, 'Trinkwasser Zugang', 'Bevölkerung'].mean().reset_index()

    for ct in water_life_continent['Kontinent'].unique():
        tmp_data = water_life_continent.query('Kontinent == @ct')
        tmp_trace = {
            'x': tmp_data[y_axis].to_list(), 
            'y': tmp_data['Trinkwasser Zugang'].to_list(),
            'type': 'scatter',
            'name': ct, 
            'mode': 'markers', 
            'text': tmp_data['Kontinent'].tolist(),
            'hovertemplate': '<span style="font-weight:bold;">Kontinent: %{text}</span><br>Prozentual %{x}<br>Trinkwasser Zugang %{y}',
            'hoverlabel': {
                'bgcolor': 'black', 
                'bordercolor': continent_styles[ct]['marker']['color'] # you should probably try-except this
            },
            'legendgroup': "group3",  # this can be any string, not just "group"
            'legendgrouptitle_text': "Kontinente",
        }
        tmp_trace.update(continent_styles[ct]) # here we add additional keys to the dictionary
        traces.append(tmp_trace)

    layout = {
        'title': f'Ünterstützung von Nonprofit Organizationen im Jahr {year}',
        'xaxis': {
            'range': [df[df['Jahr'] == year][y_axis].min()-1, df[df['Jahr'] == 2015][y_axis].max()+1],
            'showgrid': False, 
            'title': f'{y_axis} [%]'
            }, 
        'yaxis': {
            'linecolor': 'green', 
            'linewidth': 2, 
            'title': 'Trinkwasser Zugang [%]',
            'mirror': True
            },
        'height': 800,
        # 'width': 1700,
        'template': 'plotly_dark',
        'legend': {'groupclick': 'toggleitem'},
    }

    figdict = {'data': traces, 'layout': layout}
    figchart = go.Figure(**figdict)
    return figchart