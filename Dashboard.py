from dash import dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import *
from prophet import Prophet
import plotly.graph_objects as go

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

covidData = pd.read_csv("C:/Users/manik/Downloads/owid_covid-19-data_latest_covid-19-world-cases-deaths-testing.csv")
covidData = covidData[
    ['iso_code', 'continent', 'location', 'total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'life_expectancy',
     'stringency_index', 'date']]
covidData['date'] = pd.to_datetime(covidData['date'])
covidData = covidData.bfill()

notCountries = ['World', 'Upper middle income', 'South America', 'Oceania', 'North America', 'Low income',
                'International', 'High income', 'French Polynesia', 'Europe', 'European Union', 'Asia',
                'Africa', 'Asia']

countries: object = np.ndarray.tolist(pd.unique(covidData['location']))

for x in countries:
    if x in notCountries:
        countries.remove(x)

app.layout = dbc.Container(
    dbc.Row([
        html.H2("Corona Virus Information, AlphaCluster"),
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Life Expectancy"),
                        dbc.CardBody(dcc.Graph(id='lee'), style={'width': '60vh', 'height': '45vh'})
                    ], className="h-100")
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Stringency Index"),
                        dbc.CardBody(dcc.Graph(id='sii'), style={'width': '60vh', 'height': '45vh'})
                    ], className="h-100")
                )], style={"height": "500px"}),
            html.Br(),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Cases Forecast"),
                        dbc.CardBody(dcc.Graph(id='fcases'))],
                        className="graphCases")),

                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Deaths Forecast"),
                        dbc.CardBody(dcc.Graph(id='fdeaths'))],
                        className="graphDeaths")),

            ], style={"height": "400px"}
            )], width=8),

        dbc.Col([
            dbc.Row(
                dbc.Card([
                    dbc.CardHeader("Select a country"),
                    dbc.CardBody(dcc.Dropdown(id='country',
                                              options=countries,
                                              value='United States',
                                              multi=False,
                                              disabled=False,
                                              clearable=True,
                                              searchable=True,
                                              placeholder='Choose a country',
                                              className='form-dropdown',
                                              style={'width': "60%"},
                                              persistence='string',
                                              persistence_type='memory'))
                ], className="h-25")
            ),

            dbc.Row(
                dbc.Card([
                    dbc.CardHeader("Pick the start date of forecast"),
                    dbc.CardBody(dcc.DatePickerSingle(id='datePicker',
                                                      min_date_allowed=date(2020, 4, 1),
                                                      max_date_allowed=date(2022, 7, 1),
                                                      initial_visible_month=date(2021, 4, 29),
                                                      date=date(2021, 4, 29),
                                                      style={'width': "60%"}))
                ], className="h-25")
            ),

            dbc.Row(
                dbc.Card([
                    dbc.CardHeader("Number of days to Predict"),
                    dbc.CardBody(dcc.Dropdown(id='predict',
                                              options=[7, 14, 21],
                                              value=14,
                                              multi=False,
                                              clearable=False,
                                              style={'width': "60%"},
                                              persistence='string',
                                              persistence_type='local'))
                ], className="h-25")
            ),
        ]),
    ],
        justify="center",
    ),
    fluid=True,
    className="mt-3",
)

@app.callback(
    [Output('fcases', 'figure'),
     Output('fdeaths', 'figure'),
     Output('sii', 'figure'),
     Output('lee', 'figure')],

    [Input('country', 'value'),
     Input('datePicker', 'date'),
     Input('predict', 'value')]
)
def build_graph(country, datePicker, predict):
    covidCountry = covidData[(covidData.location == country)]

    endPeriod = pd.to_datetime(datePicker)
    startPeriod = endPeriod - relativedelta(months=3)
    startPeriod = pd.to_datetime(startPeriod.date())
    endPeriod = pd.to_datetime(endPeriod.date())

    testEndPeriod = endPeriod + relativedelta(days=predict)
    testStartPeriod = endPeriod
    testEndPeriod = pd.to_datetime(testEndPeriod.date())

    slicingCovidCountry = covidCountry[(covidCountry['date'] > startPeriod) & (covidCountry['date'] <= endPeriod)]
    testData = covidCountry[(covidCountry['date'] > testStartPeriod) & (covidCountry['date'] <= testEndPeriod)]

    si = float(slicingCovidCountry[slicingCovidCountry['date'] == datePicker]['stringency_index'].values)

    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=si,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ))

    le = float(slicingCovidCountry[slicingCovidCountry['date'] == datePicker]['life_expectancy'].values)

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=le,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
    ))

    datesCases = slicingCovidCountry.groupby('date').sum()['new_cases'].reset_index()
    datesCases.columns = ['ds', 'y']
    datesCases['ds'] = pd.to_datetime(datesCases['ds'])
    m = Prophet(changepoint_prior_scale=0.7, growth='linear', holidays_prior_scale=0, interval_width=0.95,
                daily_seasonality=True)
    m.fit(datesCases)
    futureDates = m.make_future_dataframe(periods=predict)
    forecast = m.predict(futureDates)
    forecasted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecasted.columns = ['date', 'Forecasted_Cases', 'Lower_Limit', 'Upper_Limit']
    forecastedCovid = pd.merge(forecasted, testData, how="right", on="date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecastedCovid['date'], y=forecastedCovid['Forecasted_Cases'],
                             mode='lines',
                             name='Forecasted Cases'))
    fig.add_trace(go.Scatter(x=forecastedCovid['date'], y=forecastedCovid['new_cases'],
                         mode='lines',
                      name='Actual Cases'))
    fig.update_layout(yaxis={'title': 'Actual vs Forecasted Cases'})
    # , title={'text': 'Forecasting COVID 19 cases',
    # 'font': {'size': 28}, 'x': 0.5,
    # 'xanchor': 'center'}, height=500, width=650
    datesDeaths = slicingCovidCountry.groupby('date').sum()['new_deaths'].reset_index()
    datesDeaths.columns = ['ds', 'y']
    datesDeaths['ds'] = pd.to_datetime(datesDeaths['ds'])
    datesDeaths.tail(7)
    m = Prophet(changepoint_prior_scale=0.25, growth='linear', holidays_prior_scale=0, interval_width=0.95,
                daily_seasonality=True)
    m.fit(datesDeaths)
    futureDates = m.make_future_dataframe(periods=predict)
    forecast = m.predict(futureDates)
    forecasted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecasted.columns = ['date', 'Forecasted_Deaths', 'Lower Limit', 'Upper Limit']
    forecastedCovid = pd.merge(forecasted, testData, how="right", on="date")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=forecastedCovid['date'], y=forecastedCovid['Forecasted_Deaths'],
                              mode='lines',
                              name='Forecasted Deaths'))
    fig1.add_trace(go.Scatter(x=forecastedCovid['date'], y=forecastedCovid['new_deaths'],
                 mode='lines',
                name='Actual Deaths'))
    fig1.update_layout(yaxis={'title': 'Actual vs Forecasted Deaths'})
    # , title = {'text': 'Forecasting COVID 19 deaths',
    #          'font': {'size': 28}, 'x': 0.5,
    #         'xanchor': 'center'}, height = 500, width = 650
    return fig, fig1, fig2, fig3


if __name__ == "__main__":
    app.run_server(debug=False)
