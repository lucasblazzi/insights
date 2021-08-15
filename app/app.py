from utils.metrics import *
from utils.yahoo_finance import historical, simple_historical
from utils.charts import line_scatter, indicators, area_chart
import streamlit as st

st.set_page_config(layout='wide')

WORKDAYS = 252


@st.cache
def close_prices(_ids, start_date, end_date):
    return simple_historical(_ids, start_date, end_date)


def dashboard(portfolio_name, weights, start_date, end_date):
    st.title(portfolio_name)
    _ids = [weight["_id"] for weight in weights]
    proportions = [weight["proportion"] for weight in weights]

    close = close_prices(_ids, start_date, end_date)
    rets = portfolio_returns(returns(close), proportions)
    cum_ret = cumulative_returns(rets)

    total_rets = total_returns(cum_ret)
    annualized_rets = annualized_returns(cum_ret, WORKDAYS)

    vol = volatility(rets)
    annualized_vol = annualized_volatility(rets, WORKDAYS)
    shp = sharpe(rets)
    dd = drawdown(cum_ret)
    max_dd = dd.min()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.plotly_chart(line_scatter(close, close.columns, "Close Prices"), use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.plotly_chart(line_scatter(cum_ret, cum_ret.columns, "Cumulative Returns"), use_container_width=True)

    info_cols = st.beta_columns(2)
    info_cols[0].plotly_chart(indicators(total_rets, "Portfolio", "Total Return"), use_container_width=True)
    info_cols[1].plotly_chart(indicators(annualized_rets, "Portfolio", "Annualized Return"))

    info_cols[0].plotly_chart(indicators(vol, "Portfolio", "Volatility"), use_container_width=True)
    info_cols[1].plotly_chart(indicators(annualized_vol, "Portfolio", "Annualized Volatility"))

    st.dataframe(shp)
    info_cols[0].plotly_chart(indicators(shp, "Portfolio", "Sharpe", ""), use_container_width=True)
    info_cols[1].plotly_chart(indicators(max_dd, "Portfolio", "Maximum Drawdown"), use_container_width=True)

    st.plotly_chart(area_chart(dd, dd.columns, "Drawdown"), use_container_width=True)


portfolio_name = "Carteira Recomendada de Ações (10SIM) - JULHO 2021"
wights = ({"_id": "VALE3.SA", "proportion": 0.15},
          {"_id": "BBDC4.SA", "proportion": 0.10},
          {"_id": "RDOR3.SA", "proportion": 0.10},
          {"_id": "LREN3.SA", "proportion": 0.10},
          {"_id": "CCRO3.SA", "proportion": 0.10},
          {"_id": "WEGE3.SA", "proportion": 0.10},
          {"_id": "PSSA3.SA", "proportion": 0.10},
          {"_id": "BRDT3.SA", "proportion": 0.10},
          {"_id": "CYRE3.SA", "proportion": 0.10},
          {"_id": "OIBR3.SA", "proportion": 0.05})


dashboard(portfolio_name, wights, "2021-07-01", "2021-07-31")