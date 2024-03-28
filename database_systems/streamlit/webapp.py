import streamlit as st

import charts.charts
import data.data as data
import utils.plot_type_dropdown as ptd

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="DBS DankDaten")

st.sidebar.header("DBS `DankDaten`")
st.sidebar.subheader("Jahr auswählen")
year = st.sidebar.selectbox('2000, 2010, 2015', ptd.years)

st.sidebar.markdown("""
---
Created with ❤️ by Oliver Grun, Tharrmeehan Krishnathasan, Choekyel Nyungmartsang.
© 2023
""")

life = data.get_life_expectancy()
water = data.get_drinking_water()
infant = data.get_infant_death_rate()
switzerland = data.get_switzerland()
water_life = data.get_water_life()
water_infant = data.get_water_infant()
organization = data.get_organization()

st.markdown(f'<p style="color:#FFFFFF;font-size:24px;"><strong>Lebenserwartung.</strong></p>', unsafe_allow_html=True)
st.plotly_chart(charts.charts.get_choropleth_life(life), config={'displayModeBar': False}, use_container_width=True)

st.markdown(f'<p style="color:#FFFFFF;font-size:24px;"><strong>Trinkwasser Zugang.</strong></p>', unsafe_allow_html=True)
st.plotly_chart(charts.charts.get_choropleth_water(water), config={'displayModeBar': False}, use_container_width=True)

st.markdown(f'<p style="color:#FFFFFF;font-size:24px;"><strong>Säuglingssterberate.</strong></p>', unsafe_allow_html=True)
st.plotly_chart(charts.charts.get_choropleth_infant(infant), config={'displayModeBar': False}, use_container_width=True)

st.markdown(f'<p style="color:#FFFFFF;font-size:24px;"><strong>Trinkwasser Zugang versus Lebenserwartung.</strong></p>', unsafe_allow_html=True)
st.plotly_chart(charts.charts.get_scatter(water_life, 'Lebenserwartung', switzerland, year), config={'displayModeBar': False}, use_container_width=True)
st.plotly_chart(charts.charts.get_scatter_organization(organization, 'Lebenserwartung', switzerland, water_life, year), config={'displayModeBar': False}, use_container_width=True)

st.markdown(f'<p style="color:#FFFFFF;font-size:24px;"><strong>Trinkwasser Zugang versus Säuglingssterberate.</strong></p>', unsafe_allow_html=True)
st.plotly_chart(charts.charts.get_scatter(water_infant, 'Säuglingssterberate', switzerland, year), config={'displayModeBar': False}, use_container_width=True)
# st.plotly_chart(charts.charts.get_scatter_organization(organization, 'Säuglingssterberate', switzerland, water_infant, year), config={'displayModeBar': False}, use_container_width=True)
# funktioniert nicht, weil VIEW von organization kein Säuglingssterberate Daten hat
st.plotly_chart(charts.charts.get_scatter_water_life_country(water_life, year), config={'displayModeBar': False}, use_container_width=True)