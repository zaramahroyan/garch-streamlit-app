import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import t as student_t
import io

st.set_page_config(page_title="Risk Dept GARCH Analyzer", layout="wide")

st.title("📈 GARCH(1,1)-t Volatility & VaR Analyzer")
st.write("Upload your `returns.xlsx` file to generate the Audit-Ready Risk Report.")

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("Choose the returns.xlsx file", type="xlsx")

if uploaded_file:
    try:
        st.info("Reading and cleaning data... Please wait.")

        # --------------------------------------------------
        # 1. READ & CLEAN DATA PROPERLY (FIXED)
        # --------------------------------------------------
        raw = pd.read_excel(uploaded_file, sheet_name="Prices", header=0)

        # Rename first column to Date
        raw.rename(columns={raw.columns[0]: "Date"}, inplace=True)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw.set_index("Date", inplace=True)

        # Replace "-" with NaN and force numeric conversion
        df_prices = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")

        # Drop completely empty columns
        df_prices = df_prices.dropna(axis=1, how="all")

        # --------------------------------------------------
        # 2. CONTAINERS
        # --------------------------------------------------
        all_returns = pd.DataFrame(index=df_prices.index)
        all_stdevs = pd.DataFrame(index=df_prices.index)
        all_var_99 = pd.DataFrame(index=df_prices.index)
        model_params = []

        # --------------------------------------------------
        # 3. PROCESS ASSETS (ROBUST LOGIC)
        # --------------------------------------------------
        progress_bar = st.progress(0)
        assets = df_prices.columns

        for i, asset in enumerate(assets):
            series = df_prices[asset].dropna()

            # Start counting AFTER first valid price
            if len(series) < 100:
                progress_bar.progress((i + 1) / len(assets))
                continue

            # Log returns
            ret = 100 * np.log(series / series.shift(1)).dropna()

            if len(ret) < 100:
                progress_bar.progress((i + 1) / len(assets))
                continue

            try:
                model = arch_model(ret, vol="GARCH", p=1, q=1, dist="t")
                res = model.fit(disp="off")

                om = res.params["omega"]
                al = res.params["alpha[1]"]
                be = res.params["beta[1]"]
                nu = res.params["nu"]

                model_params.append({
                    "Asset": asset,
                    "Omega": om,
                    "Alpha": al,
                    "Beta": be,
                    "Persistence": al + be,
                    "Nu (DF)": nu
                })


                T = len(ret)
                sigma2 = np.full(T, np.nan)

                 # Place initial variance at the 100th observation
                sigma2[99] = np.var(ret.values[:100])

                r_sq = ret.values ** 2

                # Recursive GARCH from 101st observation onward
                for t in range(100, T):
                    sigma2[t] = om + al * r_sq[t - 1] + be * sigma2[t - 1]

                        

                stdev = np.sqrt(sigma2)
                t_quantile = student_t.ppf(0.01, nu)

                all_returns[asset] = ret
                all_stdevs[asset] = pd.Series(stdev, index=ret.index)
                all_var_99[asset] = pd.Series(t_quantile * stdev, index=ret.index)

            except Exception:
                pass

            progress_bar.progress((i + 1) / len(assets))

        # --------------------------------------------------
        # 4. PREPARE EXCEL IN MEMORY
        # --------------------------------------------------
        output = io.BytesIO()

        df_prices.index = pd.to_datetime(df_prices.index)
        all_returns.index = pd.to_datetime(all_returns.index)
        all_stdevs.index = pd.to_datetime(all_stdevs.index)
        all_var_99.index = pd.to_datetime(all_var_99.index)

        with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yy") as writer:
            workbook = writer.book
            date_format = workbook.add_format({"num_format": "dd-mmm-yy"})
            blue_fmt = workbook.add_format({"bg_color": "#BDD7EE", "bold": True, "border": 1})

            def save_fmt(df, name, high=False):
                df.to_excel(writer, sheet_name=name)
                ws = writer.sheets[name]
                ws.set_column("A:A", 18, date_format)
                if high:
                    ws.set_row(101, None, blue_fmt)

            save_fmt(df_prices, "Original_Prices")
            save_fmt(all_returns, "Returns_Scaled")
            save_fmt(all_stdevs, "GARCH_Stdev", True)
            save_fmt(all_var_99, "GARCH_VaR", True)
            pd.DataFrame(model_params).to_excel(
                writer, sheet_name="Model_Parameters", index=False
            )

        st.success("✅ Analysis Complete!")

        # --------------------------------------------------
        # 5. DOWNLOAD BUTTON
        # --------------------------------------------------
        st.download_button(
            label="Download GARCH Analysis Report",
            data=output.getvalue(),
            file_name="Risk_Dept_Final_GARCH_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error: {e}")
