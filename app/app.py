from utils.metrics import Product, Portfolio
from utils.yahoo_finance import historical, simple_historical
from utils.charts import line_scatter, indicators, area_chart, bar_chart, correlation_matrix, efficient_frontier_plot
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout='wide')

WORKDAYS = 252


@st.cache
def close_prices(_ids, start_date, end_date):
    return simple_historical(_ids, start_date, end_date)


def dashboard(proportions, start_date, end_date):
    st.title("Portfolio Management")
    _ids = [p["_id"] for p in proportions]
    weights = np.array([p["proportion"] for p in proportions])
    close = close_prices(_ids, start_date, end_date)

    products = Product(close)
    portfolio = Portfolio(close, weights)

    cum_ret = products.cumulative_returns()
    portfolio_cum_ret = portfolio.cumulative_returns()
    portfolio_ret = portfolio.returns()
    portfolio_dd = portfolio.drawdown()
    portfolio_total_rets = portfolio.total_returns()
    portfolio_vol = portfolio.volatility()
    portfolio_max_dd = portfolio_dd.min()

    portfolio_annualized_rets = portfolio.annualized_returns()
    portfolio_annualized_vol = portfolio.annualized_volatility()
    portfolio_shp = portfolio.sharpe()
    portfolio_corr = products.correlation()
    portfolio_ef = portfolio.efficient_frontier()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Basic Portfolio Time Series Measures")
    g_cols = st.beta_columns(2)
    g_cols[0].plotly_chart(line_scatter(close, close.columns, "Close Prices"), use_container_width=True)
    g_cols[1].plotly_chart(line_scatter(cum_ret*100, cum_ret.columns, "Cumulative Returns"), use_container_width=True)
    g_cols[0].plotly_chart(bar_chart(portfolio_ret*100, [0], "Portfolio Daily Returns"), use_container_width=True)
    g_cols[1].plotly_chart(line_scatter(portfolio_cum_ret*100, [0], "Portfolio Cumulative Returns"), use_container_width=True)
    g_cols[0].plotly_chart(area_chart(portfolio_dd*100, [0], "Portfolio Drawdown"), use_container_width=True)
    g_cols[1].plotly_chart(correlation_matrix(portfolio_corr), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Basic Portfolio Macro Measures")
    info_cols = st.beta_columns(6)
    info_cols[0].plotly_chart(indicators(portfolio_total_rets, title="Total Return"), use_container_width=True)
    info_cols[1].plotly_chart(indicators(portfolio_vol, title="Volatility"), use_container_width=True)
    info_cols[2].plotly_chart(indicators(portfolio_annualized_rets, title="Annualized Return"), use_container_width=True)
    info_cols[3].plotly_chart(indicators(portfolio_annualized_vol, title="Annualized Volatility"), use_container_width=True)
    info_cols[4].plotly_chart(indicators(portfolio_max_dd, title="Maximum Drawdown"), use_container_width=True)
    info_cols[5].plotly_chart(indicators(portfolio_shp, title="Sharpe", suffix=""), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("Markowitz Portfolio Analysis")
    st.plotly_chart(efficient_frontier_plot(portfolio_ef), use_container_width=True)

    for k, v in portfolio_ef.items():
        if isinstance(v, list):
            comp = [f"{ticker}: {round(weight * 100, 2)}%" for ticker, weight in zip(_ids, v)]
            st.markdown(f"**{k} -** {comp}")


weights = ({"_id": "BABA34.SA", "proportion": 0.05},
           {"_id": "GOGL34.SA", "proportion": 0.10},
           {"_id": "ITSA4.SA", "proportion": 0.20},
           {"_id": "MELI34.SA", "proportion": 0.15},
           {"_id": "MUTC34.SA", "proportion": 0.05},
           {"_id": "PSSA3.SA", "proportion": 0.15},
           {"_id": "BTCR11.SA", "proportion": 0.05},
           {"_id": "BTLG11.SA", "proportion": 0.05},
           {"_id": "IRDM11.SA", "proportion": 0.05},
           {"_id": "MXRF11.SA", "proportion": 0.15},)


dashboard(weights, "2020-01-01", "2020-12-31")