import streamlit as st
from propelauth import auth
from multi_agent_stock_analysis import coordinator_agent

st.set_page_config(
    page_title="Financial Recommendation & Analysis System With Agentic AI",
    page_icon=":moneybag:",
    layout="wide",
)
st.header("Financial Recommendation & Analysis System With Agentic AI")

if st.user.is_logged_in:
    user = auth.get_user(st.user.sub)
    if user:
        st.sidebar.page_link(auth.get_account_url(), label="Account", icon="🔑")
        st.sidebar.button(
            "Logout",
            on_click=auth.log_out,
            args=(st.user.sub,),
            use_container_width=True,
        )
        st.write(f"Welcome **{user.email}** :sunglasses:")
        st.markdown(
            """
            This is a financial recommendation and analysis system powered by Agentic AI.\n
            This system can perform various tasks such as:\n
            - Analysis of individual stocks\n
            - Comparison of multiple stocks\n
            - Portfolio recommendations\n
            """
        )
        user_prompt = st.text_area(
            label="Enter your query here",
            placeholder="e.g. Compare TCS and INFOSYS stocks",
            label_visibility="collapsed",
        )
        submit_btn = st.empty()
        if submit_btn.button("Submit", key="submit_btn"):
            if user_prompt:
                submit_btn.button("Submit", disabled=True)
                result_container = st.empty()
                result_container.markdown("")
                final_prompt = user_prompt + ". All stocks are from NSE India."
                with st.spinner("Processing your request..."):
                    try:
                        response = coordinator_agent.run(final_prompt).content
                        result_container.markdown(response)
                    except Exception as e:
                        st.error(
                            "Sorry! An error occurred while processing your request. Please try again later."
                        )
                    finally:
                        submit_btn.button("Submit", disabled=False, key="submit_btn1")
            else:
                st.info("Please enter a query to get started.")
else:
    st.write("You are not logged in.")
    st.button("Login", on_click=st.login)
    st.stop()
