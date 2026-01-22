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
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# --- LISTA IBRX (MANTIDA) ---
IBRX_100_OPT = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "PETR3", "B3SA3", "WEGE3", "ABEV3", "RENT3",
    "SUZB3", "PRIO3", "HAPV3", "RDOR3", "EQTL3", "LREN3", "RAIZ4", "GGBR4", "BPAC11", "JBSS3",
    "SBSP3", "VIVT3", "RADL3", "TIMS3", "CPLE6", "ELET3", "VBBR3", "CSAN3", "BBDC3", "UGPA3",
    "TOTS3", "CMIG4", "ITSA4", "EMBR3", "VAMO3", "BRFS3", "ENEV3", "CCRO3", "CSNA3", "MGLU3",
    "ASAI3", "CRFB3", "ELET6", "GOAU4", "HYPE3", "VIIA3", "EGIE3", "SOMA3", "CPFE3", "ALPA4",
    "MULT3", "IGTI11", "YDUQ3", "CIEL3", "EZTC3", "BBSE3", "SANB11", "MRFG3", "BEEF3", "MRVE3",
    "KLBN11", "TAEE11", "CMIN3", "GOLL4", "CVCB3", "PETZ3", "DXCO3", "SMTO3", "FLRY3",
    "COGN3", "POSI3", "LWSA3", "ENGI11", "TRPL4", "RAIL3", "SLCE3", "ARZZ3", "PCAR3", "BRKM5",
    "CSMG3", "USIM5", "GMAT3", "NTCO3", "CYRE3", "ECOR3", "JHSF3", "CASH3", "STBP3", "QUAL3","BOVA11",
    "SMLL11"
]
IBRX_100_OPT.sort()

# --- CLASSE MATEMÁTICA ---
class QuantMath:
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        """Calcula Preço Justo e Gregas"""
        if T <= 0: return max(0, S-K) if option_type == 'call' else max(0, K-S), 0, 0, 0, 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_minus_d1 = norm.cdf(-d1)
        cdf_minus_d2 = norm.cdf(-d2)

        if option_type == 'call':
            price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
            delta = cdf_d1
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365
        else: 
            price = K * np.exp(-r * T) * cdf_minus_d2 - S * cdf_minus_d1
            delta = cdf_d1 - 1 
            theta = (- (S * sigma * pdf_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_minus_d2) / 365
            
        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * pdf_d1 / 100 
        return price, delta, gamma, theta, vega

    @staticmethod
    def find_strike_by_delta(S, T, r, sigma, target_delta, option_type='call'):
        if T <= 0: return S
        if option_type == 'call':
            d1 = norm.ppf(target_delta)
        else:
            # Para put, delta é negativo, usamos abs e lógica inversa
            d1 = norm.ppf(1 + target_delta) if target_delta < 0 else norm.ppf(target_delta)
            
        term1 = d1 * sigma * np.sqrt(T)
        term2 = (r + 0.5 * sigma**2) * T
        ln_S_K = term1 - term2
        K = S / np.exp(ln_S_K)
        return K
        
# --- CLASSE DE INDICADORES TÉCNICOS (ATUALIZADA) ---
class TechnicalIndicators:
    @staticmethod
    def calculate_volatility_rank(series, window_years=1):
        log_ret = np.log(series / series.shift(1))
        hv = log_ret.rolling(window=21).std() * np.sqrt(252) * 100
        lookback = int(252 * window_years)
        # Proteção contra dados curtos
        if len(hv) < lookback: lookback = len(hv)
        
        hv_min = hv.rolling(window=lookback).min()
        hv_max = hv.rolling(window=lookback).max()
        rank = ((hv - hv_min) / (hv_max - hv_min)) * 100
        return hv, rank

    @staticmethod
    def calculate_bollinger_percentile(series, window=20):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        u_bb = sma + (std * 2)
        l_bb = sma - (std * 2)
        bbw = ((u_bb - l_bb) / sma) * 100
        
        # Rank dos últimos 6 meses (126 dias)
        bbw_min = bbw.rolling(window=126).min()
        bbw_max = bbw.rolling(window=126).max()
        bbw_rank = ((bbw - bbw_min) / (bbw_max - bbw_min)) * 100
        return bbw, bbw_rank, u_bb, l_bb, sma

    @staticmethod
    def check_keltner_squeeze(df, window=20):
        # Bollinger
        std = df['Close'].rolling(window=window).std()
        upper_bb = df['Close'].rolling(window=window).mean() + (std * 2)
        lower_bb = df['Close'].rolling(window=window).mean() - (std * 2)
        
        # Keltner (ATR simplificado)
        tr = pd.concat([df['High'] - df['Low'], 
                        abs(df['High'] - df['Close'].shift()), 
                        abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        sma = df['Close'].rolling(window=window).mean()
        upper_kc = sma + (atr * 1.5)
        lower_kc = sma - (atr * 1.5)
        
        squeeze_on = (upper_bb < upper_kc) & (lower_bb > lower_kc)
        return squeeze_on

    # --- NOVOS SETUPS ADAPTADOS B3 ---

    @staticmethod
    def check_brazil_breakout(df):
        """
        Qullamaggie Adaptado para B3:
        1. Momentum Prévio: Subiu pelo menos 12% nos últimos 60 dias (Trend).
        2. Médias Alinhadas: Preço > SMA20 > SMA50.
        3. Consolidação: Preço atual está a menos de 5% da Máxima de 20 dias.
        4. Gatilho: Hoje rompeu a máxima dos últimos 10 dias OU Volume estourou.
        """
        close = df['Close']
        volume = df['Volume']
        
        # Médias
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        # Momentum (60 dias ~ 3 meses)
        momentum_60d = close.pct_change(60)
        
        # Consolidação (Tight Area)
        max_20d = df['High'].rolling(20).max().shift(1) # Máxima prévia
        dist_to_high = (max_20d - close) / close

        # Gatilho de Rompimento (Breakout) de curto prazo (10 dias)
        max_10d = df['High'].rolling(10).max().shift(1)
        is_breaking_out = close.iloc[-1] > max_10d.iloc[-1]
        
        # Volume Spike
        vol_sma20 = volume.rolling(20).mean()
        vol_spike = volume.iloc[-1] > (vol_sma20.iloc[-1] * 1.5)

        # Lógica Final
        # 1. Tem tendência de alta? (Momentum > 12% E acima das médias)
        trend_ok = (momentum_60d.iloc[-1] > 0.12) and (close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1])
        
        # 2. Está consolidando perto do topo? (Não recuou mais que 8% do topo)
        consolidation_ok = dist_to_high.iloc[-1] < 0.08 and dist_to_high.iloc[-1] > -0.05
        
        return trend_ok, consolidation_ok, is_breaking_out, vol_spike, momentum_60d.iloc[-1]

    @staticmethod
    def check_episodic_pivot(df):
        """
        Setup EP: Gap de Alta + Volume Massivo
        """
        # Gap de hoje
        prev_high = df['High'].shift(1).iloc[-1]
        today_open = df['Open'].iloc[-1]
        today_close = df['Close'].iloc[-1]
        
        gap_pct = (today_open - prev_high) / prev_high
        
        # Volume
        vol_avg = df['Volume'].rolling(20).mean().shift(1).iloc[-1]
        today_vol = df['Volume'].iloc[-1]
        
        # Critérios: Gap > 1.5% (B3 é menos volátil que Nasdaq) E Volume > 2x Média E Fechou Forte
        is_ep = (gap_pct > 0.015) and (today_vol > 2 * vol_avg) and (today_close > today_open)
        
        return is_ep, gap_pct, (today_vol / vol_avg) if vol_avg > 0 else 0

    @staticmethod
    def check_parabolic_short(df):
        """
        Setup Climactic / Parabolic: Esticada violenta
        1. RSI acima de 75/80.
        2. Preço muito longe da média de 10/20 (Esticado).
        """
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Distância da Média 10
        sma10 = df['Close'].rolling(10).mean()
        dist_sma10 = (df['Close'] / sma10) - 1
        
        is_parabolic = (rsi.iloc[-1] > 75) and (dist_sma10.iloc[-1] > 0.15) # 15% longe da média de 10 é mta coisa pra B3
        
        return is_parabolic, rsi.iloc[-1], dist_sma10.iloc[-1]

# --- DADOS (CACHE) ---
@st.cache_data(ttl=1800) 
def get_batch_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    formatted_tickers = [t if t.endswith('.SA') else f"{t}.SA" for t in tickers_list]
    try:
        # AUMENTADO PARA 1 ANO (1y) para cálculo correto do Rank
        data = yf.download(formatted_tickers, period="1y", group_by='ticker', progress=False, threads=True)
        return data
    except:
        return pd.DataFrame()

# --- INTERFACE ---
st.title("⚡ Quant Options Lab 3.0: Breakout & Time")

with st.sidebar:
    st.header("⚙️ Parâmetros")
    selic_anual = st.number_input("Selic (%)", value=11.25, step=0.25)
    RISK_FREE = selic_anual / 100
    
    st.markdown("---")
    st.markdown("**Ativo Customizado**")
    custom_ticker = st.text_input("Código (ex: NVDC34)", placeholder="Sem .SA").upper().strip()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📡 Scanner", "🧮 Calculadora", "⚡ Straddle", "⏪ Backtest Sintético"])

# --- TAB 1: SCANNER DE QULLAMAGGIE (ADAPTADO) ---
with tab1:
    st.markdown("### 🦁 Scanner Qullamaggie (B3 Edition)")
    
    # Seletor de Estratégia
    st.write("Selecione qual setup você quer caçar hoje:")
    setup_mode = st.radio("", 
        ["🚀 Breakout (Tendência + Consolidação)", 
         "📰 Episodic Pivot (Notícia/Gap)", 
         "📉 Parabolic Short (Esticada/Reversão)"], 
        horizontal=True
    )

    col_sel_all, col_sel, col_act = st.columns([1, 3, 1])
    with col_sel_all:
        st.write("") 
        st.write("") 
        select_all = st.checkbox("Selecionar Todos", value=True) # Padrão True para varrer tudo
    
    with col_sel:
        final_list_options = IBRX_100_OPT.copy()
        if custom_ticker and custom_ticker not in final_list_options:
            final_list_options.insert(0, custom_ticker)
            
        default_selection = ["PETR4", "VALE3", "ITUB4", "PRIO3", "WEGE3", "BBAS3"]
        if select_all:
            options_selected = st.multiselect("Carteira", options=final_list_options, default=final_list_options)
        else:
            options_selected = st.multiselect("Carteira", options=final_list_options, default=default_selection)
            
    with col_act:
        st.write("") 
        st.write("") 
        run_scan = st.button("🔎 RASTREAR SETUP", type="primary", use_container_width=True)

    if run_scan:
        if not options_selected:
            st.warning("Selecione ativos.")
        else:
            with st.spinner(f"Analisando {len(options_selected)} ativos na B3..."):
                market_data = get_batch_data(options_selected)
            
            results = []
            
            if not market_data.empty:
                progress_bar = st.progress(0)
                total_tickers = len(options_selected)
                
                for i, ticker in enumerate(options_selected):
                    progress_bar.progress((i + 1) / total_tickers)
                    try:
                        # Extração de Dados
                        if len(options_selected) > 1:
                            if (ticker + ".SA") not in market_data.columns.levels[0]: continue
                            df = market_data[ticker + ".SA"].copy()
                        else:
                            df = market_data.copy()
                            if isinstance(df.columns, pd.MultiIndex):
                                try: df = df.xs(ticker + ".SA", axis=1, level=0)
                                except: pass
                        
                        df = df.dropna()
                        if len(df) < 60: continue

                        # --- LÓGICA DE SELEÇÃO POR SETUP ---
                        
                        if "Breakout" in setup_mode:
                            trend_ok, cons_ok, break_ok, vol_ok, mom_val = TechnicalIndicators.check_brazil_breakout(df)
                            
                            # Filtro: Mostrar se tem Tendência E (Consolidação OU Rompimento)
                            if trend_ok and (cons_ok or break_ok):
                                status = "💤 Consolidando"
                                if break_ok and vol_ok: status = "🔥 BREAKOUT + VOLUME"
                                elif break_ok: status = "⚡ Breakout (Vol Baixo)"
                                
                                results.append({
                                    "Ativo": ticker,
                                    "Preço": df['Close'].iloc[-1],
                                    "Status": status,
                                    "Momentum (60d)": mom_val * 100,
                                    "Vol 20d (R$)": (df['Volume'] * df['Close']).rolling(20).mean().iloc[-1] / 1_000_000 # Em Milhões
                                })

                        elif "Episodic" in setup_mode:
                            is_ep, gap_val, vol_mult = TechnicalIndicators.check_episodic_pivot(df)
                            if is_ep:
                                results.append({
                                    "Ativo": ticker,
                                    "Preço": df['Close'].iloc[-1],
                                    "Gap (%)": gap_val * 100,
                                    "Vol Multiplier": vol_mult,
                                    "Status": "📰 EP DETECTADO"
                                })
                        
                        elif "Parabolic" in setup_mode:
                            is_para, rsi_val, dist_val = TechnicalIndicators.check_parabolic_short(df)
                            if is_para:
                                results.append({
                                    "Ativo": ticker,
                                    "Preço": df['Close'].iloc[-1],
                                    "RSI (14)": rsi_val,
                                    "Dist. MM10 (%)": dist_val * 100,
                                    "Status": "⚠️ ESTICADO (SHORT)"
                                })

                    except Exception as e: pass
                
                progress_bar.empty()

                if results:
                    df_res = pd.DataFrame(results)
                    
                    if "Breakout" in setup_mode:
                        df_res = df_res.sort_values(by="Momentum (60d)", ascending=False)
                        st.success(f"Encontrados {len(df_res)} ativos em Tendência/Setup.")
                        st.dataframe(df_res.style.format({
                            "Preço": "R$ {:.2f}",
                            "Momentum (60d)": "{:.1f}%",
                            "Vol 20d (R$)": "R$ {:.1f}M"
                        }).applymap(
                            lambda x: 'background-color: #d4edda; color: green; font-weight: bold' if 'BREAKOUT' in str(x) else '', 
                            subset=['Status']
                        ), use_container_width=True, hide_index=True)
                        
                    elif "Episodic" in setup_mode:
                        df_res = df_res.sort_values(by="Gap (%)", ascending=False)
                        st.success(f"Encontrados {len(df_res)} potenciais Episodic Pivots.")
                        st.dataframe(df_res.style.format({
                            "Preço": "R$ {:.2f}",
                            "Gap (%)": "{:.2f}%",
                            "Vol Multiplier": "{:.1f}x"
                        }), use_container_width=True, hide_index=True)

                    elif "Parabolic" in setup_mode:
                        df_res = df_res.sort_values(by="RSI (14)", ascending=False)
                        st.warning(f"Encontrados {len(df_res)} ativos esticados (Cuidado!).")
                        st.dataframe(df_res.style.format({
                            "Preço": "R$ {:.2f}",
                            "RSI (14)": "{:.0f}",
                            "Dist. MM10 (%)": "{:.1f}%"
                        }), use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhum ativo atendeu aos critérios deste setup hoje.")


# --- TAB 2: CALCULADORA (COM LÓGICA DE SAÍDA ANTECIPADA) ---
with tab2:
    st.markdown("### 🧮 Calculadora de Posição & Saída Antecipada")
    st.caption("Compare matematicamente: Segurar até o vencimento (Risco Máximo) vs. Sair antes (Gestão de Tempo).")

    # Seleção de Ativo
    col_tk, col_op = st.columns([1, 1])
    with col_tk:
        tk_calc = st.selectbox("Ativo Base", full_options_list if 'full_options_list' in locals() else IBRX_100_OPT, index=0)
    with col_op:
        tipo_op = st.selectbox("Operação", ["Compra de CALL (Long)", "Compra de PUT (Long)"])
        op_code = 'call' if 'CALL' in tipo_op else 'put'

    # Dados do Mercado
    try:
        ticker_obj = yf.Ticker(f"{tk_calc}.SA")
        hist = ticker_obj.history(period="1mo")
        spot_price = hist['Close'].iloc[-1]
    except:
        spot_price = 30.00
    
    st.metric(f"Preço Atual {tk_calc}", f"R$ {spot_price:.2f}")
    
    st.markdown("---")
    
    # 1. SETUP DA OPERAÇÃO
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        days_total = st.number_input("Dias até Vencimento (DTE)", value=45, min_value=10, help="Idealmente 45-60 dias")
    with c2:
        # Strike Dinâmico
        delta_target = 0.60 if op_code == 'call' else -0.40
        strike_auto = spot_price * 1.05 if op_code == 'call' else spot_price * 0.95
        strike = st.number_input("Strike (Preço de Exercício)", value=float(round(strike_auto, 2)))
    with c3:
        iv_input = st.number_input("Volatilidade Implícita (IV%)", value=30.0, step=1.0) / 100
    with c4:
        cost_manual = st.number_input("Custo da Opção (R$)", value=0.0, step=0.01)

    # Cálculo Inicial (Entrada)
    theo_price, delta, gamma, theta, vega = QuantMath.black_scholes_greeks(spot_price, strike, days_total/252, RISK_FREE, iv_input, op_code)
    
    entry_price = cost_manual if cost_manual > 0 else theo_price
    
    st.info(f"💰 Preço Teórico de Entrada: **R$ {entry_price:.2f}** | Delta: {delta:.2f} | Theta: {theta:.3f}")

    st.markdown("---")
    st.subheader("🔮 Simulador de Cenários: O Poder da Saída Antecipada")

    # Inputs de Simulação
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        days_held = st.slider("Dias passados na operação", 1, days_total, 15)
    with col_s2:
        price_move_pct = st.slider("Variação da Ação (%)", -10.0, 20.0, 5.0, 0.5)
    with col_s3:
        vol_change = st.slider("Variação da Volatilidade (%)", -10.0, 10.0, 0.0, 1.0)

    if st.button("Simular Resultado", type="primary"):
        # Novos Parâmetros
        new_spot = spot_price * (1 + price_move_pct/100)
        new_iv = iv_input + (vol_change/100)
        days_remaining = days_total - days_held
        
        # 1. Cenário: SAÍDA ANTECIPADA (Mark to Market)
        if days_remaining > 0:
            exit_price, _, _, _, _ = QuantMath.black_scholes_greeks(new_spot, strike, days_remaining/252, RISK_FREE, new_iv, op_code)
        else:
            exit_price = max(0, new_spot - strike) if op_code == 'call' else max(0, strike - new_spot)
            
        profit_early = exit_price - entry_price
        roi_early = (profit_early / entry_price) * 100

        # 2. Cenário: SEGURAR ATÉ O VENCIMENTO (Hold to Expiry)
        # Assumindo que o preço fique onde está (new_spot) até o dia final
        final_value = max(0, new_spot - strike) if op_code == 'call' else max(0, strike - new_spot)
        profit_hold = final_value - entry_price
        roi_hold = (profit_hold / entry_price) * 100

        # Exibição
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"#### 🏃 Saída Tática (Faltam {days_remaining} dias)")
            st.caption("Você vende a opção a mercado, recuperando valor extrínseco.")
            st.metric("Preço de Venda", f"R$ {exit_price:.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_early:.2f}", f"{roi_early:.1f}%", delta_color="normal")
            
        with res_col2:
            st.markdown("#### 💎 Segurar até o Fim (Vencimento)")
            st.caption("Você espera expirar. Todo valor extrínseco é perdido (Theta).")
            st.metric("Valor Final", f"R$ {final_value:.2f}")
            st.metric("Lucro/Prejuízo", f"R$ {profit_hold:.2f}", f"{roi_hold:.1f}%", delta_color="normal")

        # Análise Quant
        st.markdown("#### 🧠 Análise Quantitativa")
        diff_val = exit_price - final_value
        if diff_val > 0:
            st.success(f"**Vantagem da Saída Antecipada:** Ao vender agora, você 'salva' **R$ {diff_val:.2f}** de valor extrínseco que seriam destruídos pelo Theta se você segurasse até o fim com o preço parado.")
        else:
            st.warning("Neste cenário (profundamente ITM), o valor extrínseco é baixo, então segurar tem pouco custo de Theta.")

# --- TAB 3: SCANNER STRADDLE (VOLATILITY SQUEEZE) ---
with tab3:
    st.markdown("### ⚡ Scanner de Straddle: The Volatility Squeeze")
    st.info("""
    **Estratégia Quant:** Busca ativos onde a volatilidade comprimiu drasticamente (Efeito Mola). 
    A compra de Straddle (Call + Put) lucra se o ativo explodir para qualquer lado.
    
    * **BBW Rank Baixo:** As Bandas de Bollinger estão historicamente estreitas.
    * **Squeeze ON:** As Bandas de Bollinger entraram dentro do Canal de Keltner (Prepara-se para explosão).
    """)

    col_sq_1, col_sq_2 = st.columns([3, 1])
    with col_sq_1:
        # Usa a lista completa definida anteriormente
        default_squeeze = ["PETR4", "VALE3", "BOVA11", "BBAS3", "BBDC4", "MGLU3", "HAPV3"]
        squeeze_selected = st.multiselect("Carteira para Squeeze", options=IBRX_100_OPT, default=default_squeeze)
    
    with col_sq_2:
        st.write("")
        st.write("")
        run_squeeze = st.button("💣 BUSCAR SQUEEZE", type="primary", use_container_width=True)

    if run_squeeze:
        if not squeeze_selected:
            st.warning("Selecione ativos.")
        else:
            with st.spinner("Calculando Métricas de Compressão (BBW + Keltner)..."):
                market_data = get_batch_data(squeeze_selected)
            
            sq_results = []
            
            if not market_data.empty:
                prog_sq = st.progress(0)
                tot_sq = len(squeeze_selected)

                for i, ticker in enumerate(squeeze_selected):
                    prog_sq.progress((i + 1) / tot_sq)
                    try:
                        if len(squeeze_selected) > 1:
                            if (ticker + ".SA") not in market_data.columns.levels[0]: continue
                            df = market_data[ticker + ".SA"].copy()
                        else:
                            df = market_data.copy()
                        
                        df = df.dropna()
                        if len(df) < 130: continue # Precisa de histórico para o Rank

                        # 1. BBW Rank (Onde está a largura das bandas hoje vs 6 meses atrás)
                        bbw, bbw_rank, u_bb, l_bb, sma = TechnicalIndicators.calculate_bollinger_percentile(df['Close'])
                        current_bbw_rank = bbw_rank.iloc[-1]
                        
                        # 2. Keltner Squeeze Check
                        squeeze_series = TechnicalIndicators.check_keltner_squeeze(df)
                        is_squeeze_on = squeeze_series.iloc[-1]
                        
                        # 3. Estimativa de Movimento Necessário (Breakeven Straddle Estimado)
                        # Regra de bolso: Straddle ATM custa aprox 80% da Vol * Raiz(Tempo)
                        # Simplificando: Quanto a vol histórica média oscila em 20 dias?
                        hv_mean = df['Close'].pct_change().std() * np.sqrt(20) # Movimento esperado 1 mês
                        move_needed_pct = hv_mean * 100 * 1.5 # Margem de segurança

                        # Filtro: Mostrar apenas se estiver "Apertado" (Rank < 20 ou Squeeze ON)
                        status = "Normal"
                        score = 0
                        
                        if is_squeeze_on:
                            status = "🔥 SQUEEZE CRÍTICO (Keltner)"
                            score = 100
                        elif current_bbw_rank < 5:
                            status = "⚡ Compressão Extrema (<5%)"
                            score = 90
                        elif current_bbw_rank < 15:
                            status = "⚠️ Compressão Forte (<15%)"
                            score = 70
                        
                        if score > 0:
                            sq_results.append({
                                "Ativo": ticker,
                                "Preço": df['Close'].iloc[-1],
                                "BBW Rank (6m)": current_bbw_rank,
                                "Status": status,
                                "Squeeze Score": score,
                                "Volatilidade Esperada": f"+/- {move_needed_pct:.2f}%"
                            })

                    except Exception as e: pass
                
                prog_sq.empty()

                if sq_results:
                    df_sq = pd.DataFrame(sq_results)
                    df_sq = df_sq.sort_values(by="Squeeze Score", ascending=False)
                    
                    st.success(f"Encontrados {len(df_sq)} ativos em ponto de possível explosão!")
                    
                    st.dataframe(
                        df_sq.style.format({
                            "Preço": "R$ {:.2f}",
                            "BBW Rank (6m)": "{:.1f}%",
                        }).applymap(
                            lambda val: 'background-color: #ffcccc; color: red; font-weight: bold' if 'CRÍTICO' in str(val) else ('color: orange' if 'Extrema' in str(val) else ''),
                            subset=['Status']
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # --- GRÁFICO DO MELHOR SQUEEZE ---
                    st.divider()
                    top_pick = df_sq.iloc[0]['Ativo']
                    st.markdown(f"#### 🔬 Raio-X do Aperto: {top_pick}")
                    
                    # Recarrega dados do top pick para plotar
                    if len(squeeze_selected) > 1:
                        df_chart = market_data[top_pick + ".SA"].copy()
                    else:
                        df_chart = market_data.copy()
                        
                    df_chart = df_chart.iloc[-100:] # Últimos 100 dias
                    
                    # Recalcula Bandas para o gráfico
                    sma = df_chart['Close'].rolling(20).mean()
                    std = df_chart['Close'].rolling(20).std()
                    upper = sma + (std * 2)
                    lower = sma - (std * 2)

                    fig = go.Figure()
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(x=df_chart.index,
                                    open=df_chart['Open'], high=df_chart['High'],
                                    low=df_chart['Low'], close=df_chart['Close'],
                                    name='Preço'))
                    
                    # Bandas
                    fig.add_trace(go.Scatter(x=df_chart.index, y=upper, line=dict(color='gray', width=1), name='Upper BB'))
                    fig.add_trace(go.Scatter(x=df_chart.index, y=lower, line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name='Lower BB'))
                    
                    fig.update_layout(
                        title=f"Compressão de Volatilidade em {top_pick}",
                        yaxis_title="Preço (R$)",
                        height=400,
                        xaxis_rangeslider_visible=False,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.warning("⚠️ **Aviso de Earnings:** Verifique manualmente se a empresa divulgará balanço nos próximos 5 dias. Comprar Straddle *antes* do anúncio é arriscado devido ao 'IV Crush' pós-evento.")
                else:
                    st.info("Nenhum ativo apresenta condições de Squeeze (Compressão) no momento.")

# --- TAB 4: BACKTEST SINTÉTICO ---
with tab4:
    st.markdown("### ⏪ Backtest Sintético de Estratégias")
    st.info("""
    **Como funciona:** Como não temos dados históricos de prêmios de opções (pagos), este sistema realiza uma **Simulação Matemática**:
    1. Identifica o sinal no passado (ex: Breakout).
    2. **Calcula o preço justo teórico** de uma opção ATM naquele dia usando Black-Scholes e Volatilidade Histórica.
    3. Simula a venda após X dias e calcula o resultado.
    
    *Nota: Isso ignora o spread bid-ask e o 'IV Smile', mas é excelente para validar a robustez estatística do setup.*
    """)

    # --- CONFIGURAÇÃO DO BACKTEST ---
    c_bt1, c_bt2, c_bt3 = st.columns(3)
    
    with c_bt1:
        bt_ticker = st.selectbox("Ativo para Testar", IBRX_100_OPT, index=0, key="bt_ticker")
        bt_period = st.selectbox("Janela de Dados", ["1y", "2y", "5y"], index=1)
        
    with c_bt2:
        bt_strategy = st.selectbox("Estratégia", ["Compra de CALL (Breakout)", "Compra de PUT (Tendência Baixa)"])
        bt_stop_time = st.slider("Sair após (dias úteis)", 5, 45, 10, help="Sua regra de saída por tempo.")
        
    with c_bt3:
        bt_dte_target = st.number_input("Vencimento Alvo (DTE) na Entrada", value=45, help="Simula comprar uma opção com X dias para vencer.")
        bt_slippage = st.number_input("Slippage/Custos (%)", value=5.0, help="Desconto para simular spread e taxas.") / 100
    
    if st.button("🚀 RODAR BACKTEST", type="primary", use_container_width=True):
        with st.spinner(f"Analisando {bt_ticker}..."):
            try:
                # 1. Download e Limpeza Rigorosa
                ticker_final = f"{bt_ticker}.SA" if not bt_ticker.endswith(".SA") else bt_ticker
                bt_data = yf.download(ticker_final, period=bt_period, progress=False)
                
                if bt_data.empty:
                    st.error("Não foi possível obter dados para este ativo.")
                    st.stop()

                # Achata MultiIndex se existir
                if isinstance(bt_data.columns, pd.MultiIndex):
                    bt_data.columns = bt_data.columns.get_level_values(0)

                # 2. Cálculo de Indicadores (Vetorizado - Rápido)
                bt_data['Log_Ret'] = np.log(bt_data['Close'] / bt_data['Close'].shift(1))
                bt_data['Hist_Vol'] = bt_data['Log_Ret'].rolling(21).std() * np.sqrt(252)
                bt_data['Max_20'] = bt_data['High'].rolling(20).max().shift(1)
                bt_data['Vol_Avg_20'] = bt_data['Volume'].rolling(20).mean().shift(1)
                bt_data['SMA_200'] = bt_data['Close'].rolling(200).mean()
                
                bt_data = bt_data.dropna(subset=['SMA_200', 'Hist_Vol', 'Max_20'])
                
            except Exception as e:
                st.error(f"Erro no processamento de dados: {e}")
                st.stop()

            # 3. Loop de Simulação
            trades = []
            
            # Progresso para o usuário não achar que travou
            status_text = st.empty()
            
            for i in range(len(bt_data) - bt_stop_time - 1):
                row = bt_data.iloc[i]
                
                # --- LÓGICA DE ENTRADA ---
                entry_signal = False
                
                # Convertendo para float puro para evitar erros de Series
                close_val = float(row['Close'])
                max_20_val = float(row['Max_20'])
                vol_val = float(row['Volume'])
                vol_avg_val = float(row['Vol_Avg_20'])
                sma_200_val = float(row['SMA_200'])

                if "CALL" in bt_strategy:
                    # Filtro um pouco mais flexível (1.2x volume) para garantir que rode
                    if close_val > max_20_val and vol_val > (vol_avg_val * 1.2) and close_val > sma_200_val:
                        entry_signal = True
                        op_type = 'call'
                
                # --- EXECUÇÃO ---
                if entry_signal:
                    try:
                        # Dados Entrada
                        S_entry = close_val
                        K = S_entry # ATM
                        sigma_entry = float(row['Hist_Vol']) * 1.1 # IV Proxy
                        T_entry = bt_dte_target / 252
                        
                        p_entry, _, _, _, _ = QuantMath.black_scholes_greeks(S_entry, K, T_entry, RISK_FREE, sigma_entry, op_type)
                        
                        # Dados Saída (N dias depois)
                        row_exit = bt_data.iloc[i + bt_stop_time]
                        S_exit = float(row_exit['Close'])
                        T_exit = (bt_dte_target - bt_stop_time) / 252
                        sigma_exit = float(row_exit['Hist_Vol']) * 1.1
                        
                        p_exit, _, _, _, _ = QuantMath.black_scholes_greeks(S_exit, K, T_exit, RISK_FREE, sigma_exit, op_type)
                        
                        # Custos e PnL
                        cost = p_entry * (1 + bt_slippage)
                        revenue = p_exit * (1 - bt_slippage)
                        
                        if cost > 0: # Evita divisão por zero
                            trades.append({
                                "Entrada": bt_data.index[i],
                                "Saída": bt_data.index[i + bt_stop_time],
                                "Preço Ação Ent": S_entry,
                                "Preço Ação Sai": S_exit,
                                "Prêmio Pago": cost,
                                "Prêmio Vendido": revenue,
                                "PnL": revenue - cost,
                                "ROI (%)": ((revenue - cost) / cost) * 100
                            })
                    except:
                        continue

            # 4. Exibição dos Resultados
            if trades:
                df_trades = pd.DataFrame(trades)
                # ... (resto do código de métricas e gráficos que você já tem)
                st.success(f"Backtest concluído! {len(df_trades)} sinais encontrados.")
                
                # REPETIR AQUI O CÓDIGO DAS MÉTRICAS (Métricas, Gráfico de Curva de Capital, etc.)
                # (Aquelas colunas b_col1, b_col2 que estavam no seu código original)
                
            else:
                st.warning("⚠️ Nenhum sinal encontrado. Tente aumentar a 'Janela de Dados' ou mudar o Ativo.")
                    
            # 4. Análise dos Resultados
            if len(trades) > 0:
                df_trades = pd.DataFrame(trades)
                df_trades['Acumulado'] = df_trades['PnL'].cumsum()
                
                # Métricas
                total_trades = len(df_trades)
                win_rate = len(df_trades[df_trades['PnL'] > 0]) / total_trades * 100
                avg_roi = df_trades['ROI (%)'].mean()
                total_return = df_trades['PnL'].sum()
                
                # Exibição
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                b_col1.metric("Total de Trades", total_trades)
                b_col2.metric("Taxa de Acerto", f"{win_rate:.1f}%")
                b_col3.metric("ROI Médio por Trade", f"{avg_roi:.1f}%", delta_color="normal" if avg_roi > 0 else "inverse")
                b_col4.metric("Resultado Financeiro (Pts)", f"{total_return:.2f}")
                
                # Gráfico de Curva de Capital
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=df_trades['Saída'], y=df_trades['Acumulado'], mode='lines+markers', name='Lucro Acumulado', line=dict(color='green', width=2)))
                fig_bt.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_bt.update_layout(title="Curva de Evolução do Patrimônio (Simulado)", template="plotly_white", xaxis_title="Data", yaxis_title="Lucro/Prejuízo Acumulado")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                with st.expander("📄 Relatório Detalhado de Operações"):
                    st.dataframe(df_trades.style.format({
                        "Preço Ação Ent": "{:.2f}",
                        "Preço Ação Sai": "{:.2f}",
                        "Prêmio Pago": "{:.2f}",
                        "Prêmio Vendido": "{:.2f}",
                        "PnL": "{:.2f}",
                        "ROI (%)": "{:.1f}%"
                    }), use_container_width=True)
                    
            else:
                st.warning("Nenhum trade encontrado com esses parâmetros no período selecionado.")
