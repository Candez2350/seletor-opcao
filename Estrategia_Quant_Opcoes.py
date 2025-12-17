import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go # Nova biblioteca gráfica
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Quant Options Lab 2.0", 
    layout="wide", 
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- LISTA EXPANDIDA (IBRX100 + SELEÇÃO) ---
# Esta lista cobre a vasta maioria da liquidez da bolsa brasileira
IBRX_100_OPT = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "PETR3", "B3SA3", "WEGE3", "ABEV3", "RENT3",
    "SUZB3", "PRIO3", "HAPV3", "RDOR3", "EQTL3", "LREN3", "RAIZ4", "GGBR4", "BPAC11", "JBSS3",
    "SBSP3", "VIVT3", "RADL3", "TIMS3", "CPLE6", "ELET3", "VBBR3", "CSAN3", "BBDC3", "UGPA3",
    "TOTS3", "CMIG4", "ITSA4", "EMBR3", "VAMO3", "BRFS3", "ENEV3", "CCRO3", "CSNA3", "MGLU3",
    "ASAI3", "CRFB3", "ELET6", "GOAU4", "HYPE3", "VIIA3", "EGIE3", "SOMA3", "CPFE3", "ALPA4",
    "MULT3", "IGTI11", "YDUQ3", "CIEL3", "EZTC3", "BBSE3", "SANB11", "MRFG3", "BEEF3", "MRVE3",
    "KLBN11", "TAEE11", "CMIN3", "GOLL4", "AZUL4", "CVCB3", "PETZ3", "DXCO3", "SMTO3", "FLRY3",
    "COGN3", "POSI3", "LWSA3", "ENGI11", "TRPL4", "RAIL3", "SLCE3", "ARZZ3", "PCAR3", "BRKM5",
    "CSMG3", "USIM5", "GMAT3", "NTCO3", "CYRE3", "ECOR3", "JHSF3", "CASH3", "STBP3", "QUAL3"
]
IBRX_100_OPT.sort() # Ordenar alfabeticamente para facilitar a busca

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

# --- DADOS (CACHE) ---
@st.cache_data(ttl=1800) # Cache de 30 min
def get_batch_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    formatted_tickers = [t if t.endswith('.SA') else f"{t}.SA" for t in tickers_list]
    try:
        data = yf.download(formatted_tickers, period="6mo", group_by='ticker', progress=False, threads=True)
        return data
    except:
        return pd.DataFrame()

# --- INTERFACE ---
st.title("🦁 Quant Scanner 2.0: IBRX & Custom")

with st.sidebar:
    st.header("⚙️ Parâmetros")
    selic_anual = st.number_input("Selic (%)", value=11.25, step=0.25)
    RISK_FREE = selic_anual / 100
    
    st.markdown("---")
    st.markdown("**Adicionar Ticker Extra**")
    custom_ticker = st.text_input("Digita o código (ex: NVDC34)", placeholder="Sem .SA").upper().strip()
    
    st.info("Legenda Momentum (30d):\n🟢 > 5% (Forte)\n🌱 0 a 5% (Fraco)\n🍂 -5 a 0% (Fraco)\n🔴 < -5% (Forte)")

tab1, tab2 = st.tabs(["📡 Scanner de Mercado", "🧮 Calculadora & Gráfico"])

# --- TAB 1: SCANNER ---
with tab1:
    col_sel_all, col_sel, col_act = st.columns([1, 3, 1])
    
    with col_sel_all:
        st.write("")
        st.write("")
        select_all = st.checkbox("Selecionar Todos (Lista)", help="Analisa ~90 ativos. Pode demorar.")
    
    with col_sel:
        # Lógica de Lista: IBRX + Customizado
        final_list_options = IBRX_100_OPT.copy()
        if custom_ticker and custom_ticker not in final_list_options:
            final_list_options.insert(0, custom_ticker) # Coloca o customizado no topo
            
        default_selection = ["PETR4", "VALE3", "ITUB4", "BBAS3", "PRIO3", "WEGE3"]
        if custom_ticker: default_selection.append(custom_ticker)

        if select_all:
            options_selected = st.multiselect("Carteira", options=final_list_options, default=final_list_options)
        else:
            options_selected = st.multiselect("Carteira", options=final_list_options, default=default_selection)
            
    with col_act:
        st.write("") 
        st.write("") 
        run_scan = st.button("🔎 RASTREAR", type="primary", use_container_width=True)

    if run_scan:
        if not options_selected:
            st.warning("Selecione pelo menos um ativo.")
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
                        # TRATAMENTO ROBUSTO DE DADOS
                        if len(options_selected) > 1:
                            if (ticker + ".SA") not in market_data.columns.levels[0]: continue
                            df = market_data[ticker + ".SA"].copy()
                        else:
                            df = market_data.copy()
                            if isinstance(df.columns, pd.MultiIndex):
                                try: df = df.xs(ticker + ".SA", axis=1, level=0)
                                except: pass
                        
                        df = df.dropna()
                        if len(df) < 50: continue

                        # Indicadores
                        close = df['Close']
                        mme50 = close.ewm(span=50).mean().iloc[-1]
                        mom30 = close.pct_change(periods=30).iloc[-1] * 100
                        
                        vwap_num = (df['Close'] * df['Volume']).rolling(20).sum()
                        vwap_den = df['Volume'].rolling(20).sum()
                        vwap = (vwap_num / vwap_den).iloc[-1]
                        
                        last_price = close.iloc[-1]
                        dist_vwap = (last_price - vwap) / vwap
                        sinal = "Aguardar"
                        
                        # Lógica
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
                    
                    # Filtros
                    col_f1, col_f2 = st.columns([1,4])
                    with col_f1:
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
                        
                        # Botão de Exportação (CSV)
                        csv = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descarregar Resultados (CSV)",
                            data=csv,
                            file_name='quant_scanner_results.csv',
                            mime='text/csv',
                        )

                    else:
                        if show_only_signals: st.info("Sem sinais ativos.")
                        else: st.warning("Sem dados.")
                else:
                    st.warning("Falha nos dados.")

# --- TAB 2: CALCULADORA COM GRÁFICO INTERATIVO ---
with tab2:
    st.markdown("### 🧮 Análise Técnica & Opções")
    
    # Lista combinada para seleção
    full_options_list = IBRX_100_OPT.copy()
    if custom_ticker and custom_ticker not in full_options_list:
        full_options_list.insert(0, custom_ticker)
    
    # Se houver seleção na Tab 1, usa-a como prioridade
    if 'options_selected' in locals() and options_selected:
        current_list = options_selected
    else:
        current_list = full_options_list

    col_tk, col_op = st.columns([1, 1])
    with col_tk:
        tk_calc = st.selectbox("Selecione o Ativo", current_list, index=0)
    with col_op:
        tipo_op = st.selectbox("Operação", ["Compra de CALL", "Compra de PUT"])
        op_code = 'call' if 'CALL' in tipo_op else 'put'

    try:
        ticker_obj = yf.Ticker(f"{tk_calc}.SA")
        df_asset = ticker_obj.history(period="6mo")
        
        if not df_asset.empty:
            last_close = df_asset['Close'].iloc[-1]
            
            # Indicadores para Gráfico e Dados
            df_asset['MME50'] = df_asset['Close'].ewm(span=50).mean()
            df_asset['VWAP_Num'] = (df_asset['Close'] * df_asset['Volume']).rolling(20).sum()
            df_asset['VWAP_Den'] = df_asset['Volume'].rolling(20).sum()
            df_asset['VWAP'] = df_asset['VWAP_Num'] / df_asset['VWAP_Den']
            
            last_mme50 = df_asset['MME50'].iloc[-1]
            last_vwap = df_asset['VWAP'].iloc[-1]
            last_mom30 = df_asset['Close'].pct_change(periods=30).iloc[-1] * 100
            
            # DASHBOARD VISUAL
            st.markdown(f"**Indicadores Atuais de {tk_calc}:**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Preço", f"R$ {last_close:.2f}")
            k2.metric("MME50", f"R$ {last_mme50:.2f}", delta="Alta" if last_close > last_mme50 else "Baixa")
            k3.metric("VWAP (20d)", f"R$ {last_vwap:.2f}")
            k4.metric("Momento (30d)", f"{last_mom30:.2f}%", delta_color="normal" if last_mom30 > 0 else "inverse")
            
            # --- GRÁFICO PLOTLY (INTERATIVO) ---
            with st.expander("📈 Ver Gráfico Técnico (Candles + VWAP)", expanded=True):
                fig = go.Figure()
                
                # Candles
                fig.add_trace(go.Candlestick(x=df_asset.index,
                                open=df_asset['Open'], high=df_asset['High'],
                                low=df_asset['Low'], close=df_asset['Close'],
                                name='Preço'))
                
                # VWAP (Linha Laranja)
                fig.add_trace(go.Scatter(x=df_asset.index, y=df_asset['VWAP'], 
                                         line=dict(color='orange', width=2), name='VWAP (20d)'))
                
                # MME50 (Linha Azul)
                fig.add_trace(go.Scatter(x=df_asset.index, y=df_asset['MME50'], 
                                         line=dict(color='blue', width=1.5), name='MME50'))

                fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), 
                                  xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            spot_price = last_close 
        else:
            st.warning("Dados indisponíveis.")
            spot_price = 10.0
    except:
        spot_price = st.number_input("Cotação Manual", value=10.0)

    # 2. DEFINIÇÃO DO STRIKE (COM SLIDER)
    mode = st.radio("Definição do Strike", ["🎯 Pelo Delta (Recomendado)", "✍️ Manual"], horizontal=True)
    calc_strike = spot_price 

    if "Delta" in mode:
        st.subheader("Perfil de Risco")
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            target_delta = st.slider("Delta Alvo", 0.10, 0.90, 0.35, 0.05)
            if target_delta < 0.25: st.caption("🌶️ Pimentinha (Alto Risco)")
            elif target_delta < 0.55: st.caption("⚖️ Swing Trade (Equilibrado)")
            else: st.caption("🛡️ ITM (Conservador)")

        with col_d2: days_to_exp = st.number_input("Dias Úteis Vencimento", value=22, min_value=1)
        with col_d3: iv_est = st.number_input("IV Estimada (%)", value=30.0, step=1.0) / 100
        
        suggested_strike = QuantMath.find_strike_by_delta(spot_price, days_to_exp/252, RISK_FREE, iv_est, target_delta, op_code)
        st.info(f"📍 Delta **{target_delta:.2f}** -> Strike Sugerido: **R$ {suggested_strike:.2f}**")
        calc_strike = suggested_strike
        calc_days, calc_iv = days_to_exp, iv_est

    else: 
        cm1, cm2, cm3 = st.columns(3)
        with cm1: calc_strike = st.number_input("Strike", value=float(round(spot_price, 2)))
        with cm2: calc_days = st.number_input("Dias Úteis", value=22)
        with cm3: calc_iv = st.number_input("IV (%)", value=30.0) / 100

    # 3. VEREDITO E COMPARADOR (SECO VS TRAVA)
    st.markdown("---")
    st.subheader("🔮 Simulador: Seco vs. Trava")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        price_market = st.number_input("Preço da Opção (A Seco) no Book", value=0.0, step=0.01)
    with col_input2:
        move_pct = st.slider("Se a ação subir... (%)", 0.0, 15.0, 5.0, 0.5)

    if st.button("Simular Cenários", type="primary", use_container_width=True):
        # Cálculos A Seco
        theo_price, delta, gamma, theta, vega = QuantMath.black_scholes_greeks(spot_price, calc_strike, calc_days/252, RISK_FREE, calc_iv, op_code)
        
        # Simulação Trava (Vende 2 strikes acima aproximado)
        k_short = QuantMath.find_strike_by_delta(spot_price, calc_days/252, RISK_FREE, calc_iv, target_delta - 0.15, op_code)
        theo_price_short, _, _, _, _ = QuantMath.black_scholes_greeks(spot_price, k_short, calc_days/252, RISK_FREE, calc_iv, op_code)
        
        cost_seco = price_market if price_market > 0 else theo_price
        cost_trava = cost_seco - theo_price_short
        
        # Projeção
        future_spot = spot_price * (1 + move_pct/100)
        
        # Payoff Seco
        val_long_fut = max(0, future_spot - calc_strike)
        profit_seco = val_long_fut - cost_seco
        roi_seco = (profit_seco / cost_seco) * 100 if cost_seco > 0 else 0
        
        # Payoff Trava
        val_short_fut = max(0, future_spot - k_short)
        val_trava_fut = val_long_fut - val_short_fut
        profit_trava = val_trava_fut - cost_trava
        roi_trava = (profit_trava / cost_trava) * 100 if cost_trava > 0 else 0
        
        # Display
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown("### 🏎️ A Seco")
            st.caption(f"Custo: R$ {cost_seco:.2f}")
            st.metric("Resultado", f"R$ {profit_seco:.2f}", f"{roi_seco:.1f}%")
        
        with c_res2:
            st.markdown("### 🔒 Trava")
            st.caption(f"Custo: R$ {cost_trava:.2f} (Venda Strike {k_short:.2f})")
            st.metric("Resultado", f"R$ {profit_trava:.2f}", f"{roi_trava:.1f}%")

        if roi_trava > roi_seco + 10:
            st.info(f"💡 Para uma alta de {move_pct}%, a **Trava** é matematicamente superior.")
        elif roi_seco > roi_trava + 10:
            st.info(f"💡 Para uma alta de {move_pct}%, a **Compra a Seco** compensa o risco.")
        else:
            st.warning("Resultados similares.")
