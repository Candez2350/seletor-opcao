import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Quant Options Lab 3.0", 
    layout="wide", 
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOMIZADO ---
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 5px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- LISTA IBRX ATUALIZADA (Incluindo BOVA11 para evitar erro de default) ---
IBRX_100_OPT = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "PETR3", "B3SA3", "WEGE3", "ABEV3", "RENT3",
    "SUZB3", "PRIO3", "HAPV3", "RDOR3", "EQTL3", "LREN3", "RAIZ4", "GGBR4", "BPAC11", "JBSS3",
    "SBSP3", "VIVT3", "RADL3", "TIMS3", "CPLE6", "ELET3", "VBBR3", "CSAN3", "BBDC3", "UGPA3",
    "TOTS3", "CMIG4", "ITSA4", "EMBR3", "VAMO3", "BRFS3", "ENEV3", "CCRO3", "CSNA3", "MGLU3",
    "ASAI3", "CRFB3", "ELET6", "GOAU4", "HYPE3", "EGIE3", "CPFE3", "ALPA4",
    "MULT3", "IGTI11", "YDUQ3", "EZTC3", "BBSE3", "SANB11", "MRFG3", "BEEF3", "MRVE3",
    "KLBN11", "TAEE11", "CMIN3", "AZUL4", "CVCB3", "PETZ3", "DXCO3", "SMTO3", "FLRY3",
    "COGN3", "POSI3", "LWSA3", "ENGI11", "TRPL4", "RAIL3", "SLCE3", "ARZZ3", "PCAR3", "BRKM5",
    "CSMG3", "USIM5", "GMAT3", "NTCO3", "CYRE3", "ECOR3", "JHSF3", "STBP3", "BOVA11"
]
IBRX_100_OPT.sort()

# --- CLASSE MATEMÁTICA ---
class QuantMath:
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0: return max(0, S-K) if option_type == 'call' else max(0, K-S), 0, 0, 0, 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else: 
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1 
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * pdf_d1 / 100 
        return price, delta, gamma, theta, vega

# --- CLASSE DE INDICADORES TÉCNICOS ---
class TechnicalIndicators:
    @staticmethod
    def calculate_volatility_rank(series, window_years=1):
        log_ret = np.log(series / series.shift(1))
        hv = log_ret.rolling(window=21).std() * np.sqrt(252) * 100
        lookback = int(252 * window_years)
        hv_min = hv.rolling(window=lookback).min()
        hv_max = hv.rolling(window=lookback).max()
        rank = ((hv - hv_min) / (hv_max - hv_min)) * 100
        return hv, rank

    @staticmethod
    def check_breakout_setup(df):
        rolling_max_20 = df['High'].rolling(window=20).max().shift(1)
        vol_sma_20 = df['Volume'].rolling(window=20).mean().shift(1)
        last_close, last_vol = df['Close'].iloc[-1], df['Volume'].iloc[-1]
        is_break = last_close > rolling_max_20.iloc[-1]
        is_vol_high = last_vol > (vol_sma_20.iloc[-1] * 1.5)
        return is_break, is_vol_high, rolling_max_20.iloc[-1], vol_sma_20.iloc[-1]

    @staticmethod
    def calculate_bollinger_percentile(series, window=20):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper, lower = sma + (std * 2), sma - (std * 2)
        bbw = (upper - lower) / sma
        bbw_rank = ((bbw - bbw.rolling(126).min()) / (bbw.rolling(126).max() - bbw.rolling(126).min())) * 100
        return bbw, bbw_rank, upper, lower, sma

    @staticmethod
    def check_keltner_squeeze(df, window=20):
        # Bollinger Bands
        std = df['Close'].rolling(window).std()
        upper_bb = df['Close'].rolling(window).mean() + (std * 2)
        lower_bb = df['Close'].rolling(window).mean() - (std * 2)
        # Keltner Channels (Atr 1.5)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        ma = df['Close'].rolling(window).mean()
        upper_kc, lower_kc = ma + (atr * 1.5), ma - (atr * 1.5)
        return (upper_bb < upper_kc) & (lower_bb > lower_kc)

# --- DADOS ---
@st.cache_data(ttl=1800) 
def get_batch_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    formatted = [t if t.endswith('.SA') else f"{t}.SA" for t in tickers_list]
    try:
        data = yf.download(formatted, period="1y", group_by='ticker', progress=False)
        return data
    except: return pd.DataFrame()

# --- INTERFACE PRINCIPAL ---
st.title("⚡ Quant Options Lab 3.0")

with st.sidebar:
    st.header("⚙️ Configurações")
    selic = st.number_input("Selic (%)", value=11.25, step=0.25)
    RISK_FREE = selic / 100
    custom_ticker = st.text_input("Ticker Extra (ex: NVDC34)", "").upper().strip()

tab1, tab2, tab3, tab4 = st.tabs(["📡 Scanner", "🧮 Calculadora", "⚡ Straddle", "⏪ Backtest"])

# --- TAB 1: SCANNER ---
with tab1:
    st.subheader("🚀 Scanner de Breakout")
    options = IBRX_100_OPT.copy()
    if custom_ticker: options.insert(0, custom_ticker)
    selected = st.multiselect("Ativos para Monitorar", options, default=["PETR4", "VALE3", "ITUB4"])
    
    if st.button("Executar Scanner"):
        data = get_batch_data(selected)
        results = []
        for t in selected:
            try:
                df = data[t + ".SA"].dropna() if len(selected) > 1 else data.dropna()
                is_break, is_vol, r_max, r_vol = TechnicalIndicators.check_breakout_setup(df)
                hv, rank = TechnicalIndicators.calculate_volatility_rank(df['Close'])
                if is_break:
                    results.append({"Ativo": t, "Preço": df['Close'].iloc[-1], "Sinal": "🔥 BREAKOUT" if is_vol else "⚠️ Sem Volume", "HV Rank": f"{rank.iloc[-1]:.1f}%"})
            except: continue
        if results: st.dataframe(pd.DataFrame(results), use_container_width=True)
        else: st.info("Nenhum sinal detectado.")

# --- TAB 2: CALCULADORA ---
with tab2:
    st.subheader("🧮 Simulação de Gregas e Saída")
    c1, c2 = st.columns(2)
    with c1:
        s_price = st.number_input("Preço Ativo Objeto (S)", value=30.0)
        k_strike = st.number_input("Strike (K)", value=31.0)
    with c2:
        dte = st.number_input("Dias para Vencimento", value=45)
        iv = st.number_input("Volatilidade Implícita (%)", value=35.0) / 100

    p, d, g, t, v = QuantMath.black_scholes_greeks(s_price, k_strike, dte/252, RISK_FREE, iv)
    st.metric("Prêmio Teórico", f"R$ {p:.2f}")
    st.write(f"**Delta:** {d:.2f} | **Theta Diário:** R$ {t:.3f} | **Vega (1%):** R$ {v:.3f}")

# --- TAB 3: STRADDLE / SQUEEZE ---
with tab3:
    st.subheader("⚡ Scanner de Squeeze (Compressão)")
    
    # Filtro de segurança para evitar o erro StreamlitAPIException
    default_sq = [t for t in ["PETR4", "VALE3", "BOVA11", "BBAS3"] if t in IBRX_100_OPT]
    sq_selected = st.multiselect("Ativos Squeeze", IBRX_100_OPT, default=default_sq)
    
    if st.button("Buscar Squeeze"):
        data_sq = get_batch_data(sq_selected)
        sq_res = []
        for t in sq_selected:
            try:
                df = data_sq[t + ".SA"].dropna() if len(sq_selected) > 1 else data_sq.dropna()
                is_sq = TechnicalIndicators.check_keltner_squeeze(df)
                _, b_rank, _, _, _ = TechnicalIndicators.calculate_bollinger_percentile(df['Close'])
                if is_sq.iloc[-1] or b_rank.iloc[-1] < 15:
                    sq_res.append({"Ativo": t, "Status": "🔥 SQUEEZE", "BBW Rank": f"{b_rank.iloc[-1]:.1f}%"})
            except: continue
        if sq_res: st.table(pd.DataFrame(sq_res))
        else: st.info("Nenhuma compressão extrema encontrada.")

# --- TAB 4: BACKTEST ---
with tab4:
    st.subheader("⏪ Backtest Sintético")
    st.write("Simulação de compra de Call em rompimentos de 20 dias com saída após 10 dias úteis.")
    bt_t = st.selectbox("Ativo Teste", IBRX_100_OPT, index=0)
    if st.button("Rodar Backtest"):
        df_bt = yf.download(f"{bt_t}.SA", period="2y", progress=False)
        # Lógica simplificada de sinais
        df_bt['Max20'] = df_bt['High'].rolling(20).max().shift(1)
        trades = []
        for i in range(20, len(df_bt)-10):
            if df_bt['Close'].iloc[i] > df_bt['Max20'].iloc[i]:
                # Compra fictícia ATM
                p_in = df_bt['Close'].iloc[i]
                p_out = df_bt['Close'].iloc[i+10]
                trades.append(p_out - p_in)
        if trades:
            st.write(f"Total de Trades: {len(trades)}")
            st.write(f"Resultado Acumulado (Pontos): {sum(trades):.2f}")
        else: st.warning("Sem trades no período.")
