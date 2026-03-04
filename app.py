import streamlit as st
import pandas as pd
# 导入核心定价引擎
from green_ammonia_lsm import GreenAmmoniaRealOptions 

# --- 页面全局配置 ---
st.set_page_config(page_title="Green Ammonia Valuation PRO", page_icon="⚡", layout="wide")

# --- 主界面标题与介绍 ---
st.title("⚡ Green Ammonia Real Options Valuation Engine")
st.markdown("### Professional investment timing model for renewable energy projects")
st.markdown("This tool utilizes the **Least Squares Monte Carlo (LSM)** algorithm to evaluate the optimal investment timing by balancing technological progress (CAPEX reduction) against market volatility.")
st.divider()

# --- 左侧控制面板 (Sidebar) ---
st.sidebar.header("⚙️ Market & Asset Parameters")

p_amm_input = st.sidebar.slider(
    "Ammonia Price (CNY/t) - 氨价", 
    min_value=2500, max_value=5000, value=3700, step=100
)

day_amm_input = st.sidebar.slider(
    "Daily Capacity (t/day) - 日产能", 
    min_value=1500, max_value=4000, value=2800, step=100
)

inv_input = st.sidebar.number_input(
    "Initial CAPEX (CNY) - 初始投资额", 
    min_value=5.0e8, max_value=3.0e9, value=1.5e9, step=1e8, format="%e"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pro Tip**: The full model supports multi-region grid price matrices and customizable stochastic volatility settings.")

# --- 执行按钮与核心逻辑 ---
if st.sidebar.button("🚀 Run Monte Carlo Simulation", type="primary"):
    
    with st.spinner('Running 5,000 Monte Carlo paths and executing backward induction... Please wait.'):
        
        # 实例化我们在上一节写好的类
        model = GreenAmmoniaRealOptions(p_amm_base=p_amm_input, day_amm_base=day_amm_input)
        
        # 执行计算
        opt_timing_prob = model.execute_lsm_valuation(initial_investment=inv_input)
        
        st.success("Simulation Complete! Optimal timing converged successfully.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Optimal Investment Timing Distribution")
            st.markdown("Probability of triggering the investment decision in specific years.")
            st.bar_chart(opt_timing_prob, color="#00ffcc")
            
        with col2:
            st.subheader("💡 Actionable Insights")
            st.metric(label="Most Likely Investment Year", value=str(opt_timing_prob.idxmax()))
            st.metric(label="Peak Probability", value=f"{opt_timing_prob.max()*100:.1f}%")
            st.markdown("""
            **Analysis:** A later optimal year suggests that the value of waiting (due to expected 8% p.a. CAPEX reductions) outweighs the immediate operating cash flows. 
            """)

        # --- 新增功能：导出 CSV 报告 ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Export Data")
        csv_data = opt_timing_prob.reset_index()
        csv_data.columns = ['Investment Year', 'Probability']
        csv = csv_data.to_csv(index=False).encode('utf-8')
        
        st.sidebar.download_button(
            label="Download Probability Distribution (.csv)",
            data=csv,
            file_name='green_ammonia_lsm_results.csv',
            mime='text/csv',
        )

else:
    st.info("👈 Please adjust the parameters on the left and click 'Run' to generate the valuation model.")

# --- 💰 商业变现：Gumroad 购买入口 ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛒 Unlock Full Source Code")
st.sidebar.markdown(
    "Get the underlying Object-Oriented Python script and the mathematical whitepaper (PDF) to build your own bespoke models."
)

# ⚠️ 注意：请将下面的链接替换为你真实的 Gumroad 商品链接！
gumroad_url = "https://green-ammonia-valuation-g8ge9raagptcyxdttvpnqn.streamlit.app/" 
st.sidebar.link_button("Buy on Gumroad ($99)", gumroad_url, type="primary")

# --- 底部信任背书 ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Developed by Quantitative Researcher @ University of Warwick <br>"
    "<em>For bespoke financial modeling or consulting, please connect on LinkedIn.</em>"
    "</div>", 
    unsafe_allow_html=True
)

