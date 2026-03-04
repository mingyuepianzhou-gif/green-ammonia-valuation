import streamlit as st
import pandas as pd
# 导入我们刚刚写好的核心定价引擎
from green_ammonia_lsm import GreenAmmoniaRealOptions 

# 设置页面配置，让它看起来像一个专业的SaaS控制台
st.set_page_config(page_title="Green Ammonia Valuation PRO", layout="wide")

# 主界面标题
st.title("⚡ Green Ammonia Real Options Valuation Engine")
st.markdown("### Professional investment timing model for renewable energy projects")
st.markdown("This tool utilizes Least Squares Monte Carlo (LSM) to evaluate the optimal investment timing by balancing technical progress (CAPEX reduction) against market volatility.")

st.divider()

# 左侧控制面板 (Sidebar)
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
st.sidebar.info("💡 **Pro Tip**: In the commercial version, you can unlock advanced volatility settings and multi-region grid price matrices.")

# 执行按钮
if st.sidebar.button("🚀 Run Monte Carlo Simulation", type="primary"):
    
    # 进度提示组件
    with st.spinner('Running 5000 Monte Carlo paths and executing backward induction... Please wait.'):
        
        # 实例化我们在上一节写好的类
        model = GreenAmmoniaRealOptions(p_amm_base=p_amm_input, day_amm_base=day_amm_input)
        
        # 执行计算
        opt_timing_prob = model.execute_lsm_valuation(initial_investment=inv_input)
        
        # 结果展示区
        st.success("Simulation Complete! Model converged successfully.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Optimal Investment Timing Distribution")
            st.markdown("Probability of triggering investment decision in specific years.")
            # 直接使用 Streamlit 自带的柱状图组件渲染 Pandas Series
            st.bar_chart(opt_timing_prob, color="#2E86C1")
            
        with col2:
            st.subheader("💡 Actionable Insights")
            st.metric(label="Most Likely Investment Year", value=str(opt_timing_prob.idxmax()))
            st.metric(label="Probability", value=f"{opt_timing_prob.max()*100:.1f}%")
            st.markdown("""
            **Analysis:** A later optimal year suggests that the value of waiting (due to expected CAPEX reductions) outweighs the immediate cash flows.
            """)
else:
    st.info("👈 Please adjust the parameters on the left and click 'Run' to generate the valuation model.")