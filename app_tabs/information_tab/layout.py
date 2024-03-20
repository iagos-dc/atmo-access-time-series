from dash import html
import dash_bootstrap_components as dbc

from app_tabs.common.layout import INFORMATION_TAB_VALUE, get_help_icon, get_next_button


PASS_INFO_BUTTON_ID = 'pass-info-button'


def _get_description_table(actris_logo, iagos_logo, icos_logo):
    row1 = html.Tr([
        html.Td(
            html.A(
                html.Img(
                    src=actris_logo,
                    style={'height': '140px', 'display': 'block', 'margin': '0 auto'}
                ),
                href="https://www.actris.eu/", target="_blank"
            )), html.Td(children=[
            html.Div(
                "ACTRIS is the pan-European research infrastructure producing high-quality data and information on short-lived atmospheric constituents and on the processes leading to the variability of these constituents in natural and controlled atmospheres. ACTRIS data from observational National Facilities means the ACTRIS variables resulting from measurements that fully comply with the standard operating procedures (SOP), measurement recommendations, and quality guidelines established within ACTRIS."),
            html.Div(children=[
                html.Span("Other data from the EBAS repository ("),
                html.A(
                    html.Span("https://ebas.nilu.no"),
                    href="https://ebas.nilu.no", target="_blank"
                ),
                html.Span(") is also available through this service."),
            ]),
            html.Div(children=[
                html.Span("More information in the ACTRIS Data Center: "),
                html.A(
                    html.Span("https://www.actris.eu/topical-centre/data-centre"),
                    href="https://www.actris.eu/topical-centre/data-centre", target="_blank"
                ),
            ]),
        ]),
        html.Td(
            "ACTRIS data are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).")])
    row2 = html.Tr([
        html.Td(
            html.A(
                html.Img(
                    src=iagos_logo,
                    style={'width': '210px', 'display': 'block', 'margin': '0 auto'}
                ),
                href="https://iagos.aeris-data.fr/", target="_blank"
            )),
        html.Td(children=[
            html.Div(
                "The IAGOS datasets available are Level 3 data products derived from Level 2 products: Final quality controlled observational data."),
            html.Div(
                "Monthly means timeseries have been calculated for all airports visited by the IAGOS fleet. Means are available within four layers: surface layer (below 500 m a.g.l.), planetary boundary layer (PBL), free troposphere (FT), upper troposphere (UT)."),
            html.Div(
                "Concentrations of Ozone, Carbon Monoxide, H2O gas and relative humidity are provided as well as meteorological fields: air pressure, air temperature and wind."),
            html.Div(children=[
                html.Span("More information on the IAGOS Data Portal: "),
                html.A(
                    html.Span("https://iagos.org"),
                    href="http://iagos.org", target="_blank"
                ),
            ]),
        ]),
        html.Td(
            "IAGOS data are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).")])
    row3 = html.Tr([
        html.Th(
            html.A(
                html.Img(
                    src=icos_logo,
                    style={'width': '210px', 'display': 'block', 'margin': '0 auto'}
                ),
                href="https://www.icos-cp.eu/", target="_blank"
            ), style={'text-align': 'center'}),
        html.Td(children=[
            html.Div(
                "ICOS data are from the atmospheric network of ICOS Research Infrastructure for 36 stations and 90 vertical levels."),
            html.Div(
                "The collection used contains the final quality controlled hourly averaged data for the mole fractions of CO2, CH4, N2O, CO and meteorological observations measured at the relevant vertical levels of the measurements stations."),
            html.Div(
                "All stations follow the ICOS Atmospheric Station specification V2.0 (doi:10.18160/GK28-218) and are certified as ICOS atmospheric stations Class I or II. Data processing has been performed as described in Hazan et al., 2016"),
            html.Div(children=[
                html.Span("More information on the ICOS Data Portal: "),
                html.A(
                    html.Span("https://data.icos-cp.eu/"),
                    href="https://data.icos-cp.eu/", target="_blank"
                ),
            ]),
        ]),
        html.Td(
            "ICOS data are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0).")])

    table_body = [html.Tbody([row1, row2, row3])]

    return dbc.Table(table_body, bordered=True)


def get_information_tab(actris_logo, iagos_logo, icos_logo):
    return dbc.Tab(
        label='0. Information',
        id=INFORMATION_TAB_VALUE,
        tab_id=INFORMATION_TAB_VALUE,
        children=[
            html.Div(
                children=[
                    dbc.Row(
                        dbc.Col(
                            children=dbc.Row(
                                [
                                    dbc.Col(get_help_icon()),
                                    dbc.Col(html.Div(get_next_button(PASS_INFO_BUTTON_ID))),
                                ],
                                align='center'
                            ),
                            width='auto',
                        ),
                        justify='end',
                        style={'margin-bottom': '10px'},
                    ),
                    html.H6('This service allows you to search, analyse and visualise data from three Atmosphere European Research Infrastructures.'),
                    html.H6('It has been implemented in the framework of the European Project ATMO-ACCESS to demonstrate interoperability within the Research Infrastructures'),
                    html.H6('Only Essential Climate Variables are available. You can find more information about the provided datasets in the table below.'),
                ],
                style={'margin-top': '5px', 'margin-left': '20px', 'margin-right': '20px'},
            ),
            _get_description_table(actris_logo, iagos_logo, icos_logo)
        ]
    )
