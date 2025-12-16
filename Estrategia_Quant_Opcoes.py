import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Quant Options Lab Pro", 
    layout="wide", 
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- LISTA IBRX ---
IBRX_OPT_FULL = [
    "PETR4", "VALE3", "BOVA11", "ITUB4", "BBDC4", "BBAS3", "WEGE3", "PRIO3", "ELET3", "GGBR4",
    "ABEV3", "RENT3", "B3SA3", "SUZB3", "JBSS3", "RAIZ4", "CSNA3", "RDOR3", "SBSP3", "EQTL3",
    "LREN3", "VIVT3", "TIMS3", "HAPV3", "RADL3", "CPLE6", "CMIG4", "UGPA3", "CSAN3", "TOTS3",
    "EMBR3", "BRFS3", "CRFB3", "MGLU3", "VIIA3", "CCRO3", "EGIE3", "GOAU4", "MULT3", "BPAC11",
    "IGTI11", "ENEV3", "CMIN3", "MRFG3", "BEEF3", "ASAI3", "HYPE3", "KLBN11", "SANB11", "TAEE11",
    "AZUL4", "GOLL4", "CVCB3", "PETZ3", "SOMA3", "ALPA4", "EZTC3", "CYRE3", "MRVE3", "JHSF3",
    "ECOR3", "DXCO3", "LWSA3", "CASH3", "PCAR3", "POSI3", "SLCE3", "SMTO3", "ARZZ3", "FLRY3"
]

# --- CLASSE MATEMÁTICA ---
class QuantMath:
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0: return 0, 0, 0, 0, 0 
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
            target_delta_neg = -abs(target_delta)
            d1 = norm.ppf(1 + target_delta_neg)
        term1 = d1 * sigma * np.sqrt(T)
        term2 = (r + 0.5 * sigma**2) * T
        ln_S_K = term1 - term2
        K = S / np.exp(ln_S_K)
        return K

# --- DADOS ---
@st.cache_data(ttl=3600)
def get_batch_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    formatted_tickers = [t if t.endswith('.SA') else f"{t}.SA" for t in tickers_list]
    try:
        data = yf.download(formatted_tickers, period="6mo", group_by='ticker', progress=False, threads=True)
        return data
    except:
        return pd.DataFrame()

# --- INTERFACE ---
st.title("🦁 Quant Scanner Pro: Momentum & Trend")

with st.sidebar:
    st.header("⚙️ Parâmetros")
    selic_anual = st.number_input("Selic (%)", value=11.25, step=0.25)
    RISK_FREE = selic_anual / 100
    st.info("Legenda Momentum (30d):\n🟢 > 5% (Forte)\n🌱 0 a 5% (Fraco)\n🍂 -5 a 0% (Fraco)\n🔴 < -5% (Forte)")

tab1, tab2 = st.tabs(["📡 Scanner de Mercado", "🧮 Laboratório de Opções"])

# --- TAB 1: SCANNER ---
with tab1:
    col_sel_all, col_sel, col_act = st.columns([1, 3, 1])
    
    with col_sel_all:
        st.write("")
        st.write("")
        select_all = st.checkbox("Selecionar Todos", help="Carrega os 70 ativos.")
    
    with col_sel:
        default_selection = ["PETR4", "VALE3", "ITUB4", "BBAS3", "PRIO3", "WEGE3"]
        if select_all:
            options_selected = st.multiselect("Carteira", options=IBRX_OPT_FULL, default=IBRX_OPT_FULL)
        else:
            options_selected = st.multiselect("Carteira", options=IBRX_OPT_FULL, default=default_selection)
            
    with col_act:
        st.write("") 
        st.write("") 
        run_scan = st.button("🔎 RASTREAR", type="primary", use_container_width=True)

    if run_scan:
        if not options_selected:
            st.warning("Selecione ativos.")
        else:
            with st.spinner(f"Processando {len(options_selected)} ativos..."):
                market_data = get_batch_data(options_selected)
            
            results = []
            
            if not market_data.empty:
                progress_bar = st.progress(0)
                total_tickers = len(options_selected)
                
                for i, ticker in enumerate(options_selected):
                    progress_bar.progress((i + 1) / total_tickers)
                    try:
                        if len(options_selected) > 1:
                            if (ticker + ".SA") not in market_data.columns.levels[0]: continue
                            df = market_data[ticker + ".SA"].copy()
                        else:
                            df = market_data.copy()
                            if isinstance(df.columns, pd.MultiIndex):
                                try:
                                    df = df.xs(ticker + ".SA", axis=1, level=0)
                                except: pass
                        
                        df = df.dropna()
                        if len(df) < 50: continue

                        close = df['Close']
                        mme50 = close.ewm(span=50).mean().iloc[-1]
                        mom30 = close.pct_change(periods=30).iloc[-1] * 100
                        
                        vwap_num = (df['Close'] * df['Volume']).rolling(20).sum()
                        vwap_den = df['Volume'].rolling(20).sum()
                        vwap = (vwap_num / vwap_den).iloc[-1]
                        
                        last_price = close.iloc[-1]
                        
                        dist_vwap = (last_price - vwap) / vwap
                        sinal = "Aguardar"
                        
                        if last_price > mme50:
                            if -0.015 <= dist_vwap <= 0.015: 
                                if mom30 > 0: sinal = "COMPRA"
                                else: sinal = "Aguardar (Sem Mom.)"
                        elif last_price < mme50:
                            if -0.015 <= dist_vwap <= 0.015: 
                                if mom30 < 0: sinal = "VENDA"
                                else: sinal = "Aguardar (Sem Mom.)"

                        results.append({
                            "Ativo": ticker,
                            "Preço": last_price,
                            "VWAP": vwap,
                            "MME50": mme50,
                            "Mom30d": mom30,
                            "Sinal": sinal
                        })
                    except: pass 
                
                progress_bar.empty()

                if results:
                    df_res = pd.DataFrame(results)
                    
                    col_filter1, col_filter2 = st.columns([1,4])
                    with col_filter1:
                        show_only_signals = st.checkbox("Apenas Oportunidades", value=True)
                    
                    if show_only_signals:
                        df_final = df_res[df_res["Sinal"].str.contains("COMPRA|VENDA")].copy()
                    else:
                        df_final = df_res.copy()

                    if not df_final.empty:
                        df_final['Mom_Abs'] = df_final['Mom30d'].abs()
                        df_final = df_final.sort_values(by="Mom_Abs", ascending=False).drop(columns=['Mom_Abs'])

                        def style_momentum(val):
                            if val >= 5: return 'color: #007000; font-weight: bold' 
                            elif 0 <= val < 5: return 'color: #6B8E23'
                            elif -5 < val < 0: return 'color: #CD5C5C'
                            else: return 'color: #bd0000; font-weight: bold'

                        st.dataframe(
                            df_final.style.format({
                                "Preço": "R$ {:.2f}",
                                "VWAP": "R$ {:.2f}",
                                "MME50": "R$ {:.2f}",
                                "Mom30d": "{:.2f}%"
                            }).applymap(
                                lambda val: 'background-color: #d4edda; color: green; font-weight: bold' if 'COMPRA' in str(val) else ('background-color: #f8d7da; color: red; font-weight: bold' if 'VENDA' in str(val) else ''), 
                                subset=['Sinal']
                            ).applymap(
                                style_momentum,
                                subset=['Mom30d']
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        if show_only_signals:
                            st.info("Nenhuma oportunidade ativa. Desmarque o filtro para ver tudo.")
                        else:
                            st.warning("Nenhum dado retornado.")
                else:
                    st.warning("Falha ao processar dados.")

# --- TAB 2: CALCULADORA COM DASHBOARD ATIVO ---
with tab2:
    st.markdown("### 🧮 Laboratório de Opções")
    
    calc_options = options_selected if 'options_selected' in locals() and options_selected else IBRX_OPT_FULL
    
    col_tk, col_op = st.columns([1, 1])
    with col_tk:
        tk_calc = st.selectbox("Selecione o Ativo", calc_options, index=0)
    with col_op:
        tipo_op = st.selectbox("Operação", ["Compra de CALL", "Compra de PUT"])
        op_code = 'call' if 'CALL' in tipo_op else 'put'

    try:
        ticker_obj = yf.Ticker(f"{tk_calc}.SA")
        df_asset = ticker_obj.history(period="6mo")
        
        if not df_asset.empty:
            last_close = df_asset['Close'].iloc[-1]
            last_mme50 = df_asset['Close'].ewm(span=50).mean().iloc[-1]
            last_mom30 = df_asset['Close'].pct_change(periods=30).iloc[-1] * 100
            
            v_num = (df_asset['Close'] * df_asset['Volume']).rolling(20).sum()
            v_den = df_asset['Volume'].rolling(20).sum()
            last_vwap = (v_num / v_den).iloc[-1]
            
            st.markdown(f"**Indicadores Atuais de {tk_calc}:**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Preço", f"R$ {last_close:.2f}")
            k2.metric("Tendência (MME50)", f"R$ {last_mme50:.2f}", delta_color="normal", 
                      delta="Alta" if last_close > last_mme50 else "Baixa")
            k3.metric("VWAP (20d)", f"R$ {last_vwap:.2f}")
            k4.metric("Momento (30d)", f"{last_mom30:.2f}%", 
                      delta_color="normal" if last_mom30 > 0 else "inverse")
            
            st.divider()
            spot_price = last_close 
        else:
            st.warning("Dados indisponíveis.")
            spot_price = 10.0
    except:
        spot_price = st.number_input("Cotação Manual", value=10.0)

    # 2. DEFINIÇÃO DO STRIKE
    mode = st.radio("Definição do Strike", ["🎯 Pelo Delta (Recomendado)", "✍️ Manual"], horizontal=True)
    calc_strike = spot_price 

    if "Delta" in mode:
        st.subheader("Perfil de Risco")
        
        # --- MUDANÇA AQUI: VOLTANDO PARA O SLIDER NUMÉRICO ---
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            target_delta = st.slider(
                "Delta Alvo", 
                min_value=0.10, 
                max_value=0.90, 
                value=0.35, 
                step=0.05,
                help="0.20 (Arriscado) | 0.35 (Equilibrado) | 0.70 (Conservador)"
            )
            
            # Legenda Dinâmica para ajudar na escolha
            if target_delta < 0.25:
                st.caption("🌶️ Perfil: Pimentinha (Alto Risco)")
            elif target_delta < 0.55:
                st.caption("⚖️ Perfil: Swing Trade (Equilibrado)")
            else:
                st.caption("🛡️ Perfil: Substituto da Ação (ITM)")

        with col_d2: 
            days_to_exp = st.number_input("Dias Úteis Vencimento", value=22, min_value=1)
        with col_d3: 
            iv_est = st.number_input("IV Estimada (%)", value=30.0, step=1.0) / 100
        
        suggested_strike = QuantMath.find_strike_by_delta(spot_price, days_to_exp/252, RISK_FREE, iv_est, target_delta, op_code)
        
        st.info(f"📍 Para Delta **{target_delta:.2f}**, procure o Strike próximo de: **R$ {suggested_strike:.2f}**")
        calc_strike = suggested_strike
        calc_days, calc_iv = days_to_exp, iv_est

    else: 
        cm1, cm2, cm3 = st.columns(3)
        with cm1: calc_strike = st.number_input("Strike", value=float(round(spot_price, 2)))
        with cm2: calc_days = st.number_input("Dias Úteis", value=22)
        with cm3: calc_iv = st.number_input("IV (%)", value=30.0) / 100

    # 3. VEREDITO
    st.markdown("---")
    col_res_in, col_res_out = st.columns([1, 2])
    with col_res_in:
        price_market = st.number_input("Preço Book (R$)", value=0.0, step=0.01)
        btn_calc = st.button("Calcular Veredito", type="primary", use_container_width=True)

    if btn_calc:
        theo_price, delta, gamma, theta, vega = QuantMath.black_scholes_greeks(spot_price, calc_strike, calc_days/252, RISK_FREE, calc_iv, op_code)
        with col_res_out:
            met1, met2, met3 = st.columns(3)
            met1.metric("Preço Justo", f"R$ {theo_price:.2f}")
            met2.metric("Delta", f"{delta:.2f}")
            met3.metric("Theta", f"R$ {theta:.3f}")
            if price_market > 0:
                diff = (price_market - theo_price) / theo_price * 100
                if diff > 10: st.error(f"🚨 Cara (+{diff:.1f}%)")
                elif diff < -10: st.success(f"💎 Barata ({diff:.1f}%)")
                else: st.warning(f"⚖️ Justa ({diff:.1f}%)")