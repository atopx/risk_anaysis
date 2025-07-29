"""
基于Streamlit的风险数据分析应用
==============================

本程序通过Streamlit框架构建一个简单的数据分析平台，用于对上传的Excel数据表进行风险分析。

主要功能包含：

1. **数据加载与预览**：允许用户上传Excel文件（扩展名为 ``.xlsx`` 或 ``.xls``），程序自动读取表格并显示数据概览。根据Streamlit官方文档，``st.file_uploader``
   会将上传的文件内容以 ``BytesIO`` 的形式提供给程序，开发者可以直接用 ``pandas`` 读取【409715982964876†L135-L158】。

2. **等级占比分析**：计算并展示"参考等级(电网)"、"风险评估等级"、"风险分析等级"的占比情况。通过对三种等级字段进行 ``melt``
   操作后统计分布，并利用Altair绘制柱状图。这一部分可以帮助理解各等级间的差异和一致性。

3. **数值指标与参考等级的关系**：展示综合风险值及各维度风险值与电网参考等级的平均值，使用Altair绘制分组柱状图；同时提供散点图，
   将数值指标与综合风险值或动态因子关联，可通过颜色编码展示不同等级之间的差异。Altair的官方示例说明，
   给散点图添加颜色编码只需在 ``encode`` 中将一个分类变量映射到 ``color``，这种色彩映射可以帮助区分不同类别【250448223279343†L80-L87】。

4. **动态阈值划分风险等级**：提供滑动条让用户调整综合风险值的低/中/高阈值。根据阈值划分生成虚拟的"风险等级"列，并与电网参考等级对比；
   展示两者占比差异和混淆矩阵，帮助评估现有模型与参考标准的偏差。

5. **补充分析**：包括各数值指标之间的相关性热图、不同作业性质或是否跃迁对风险值的影响等，以挖掘数据中潜在的规律。

要在本地运行该程序，请确保安装 ``streamlit``、``pandas``、``numpy`` 和 ``altair``。

运行示例：

```bash
streamlit run risk_analysis_app.py
```
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


# 定义等级排序
RISK_LEVEL_ORDER = ["可接受", "低", "中", "高"]


def load_data(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """将上传的Excel文件读取为DataFrame。

    Args:
        uploaded_file: Streamlit的上传对象。
    Returns:
        pandas DataFrame。
    """
    # 通过pandas读取Excel. 在Streamlit环境中，uploaded_file为BytesIO对象，可以直接传入
    df = pd.read_excel(uploaded_file)
    return df


def compute_level_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """统计参考等级、评估等级和分析等级的占比。

    Returns:
        包含等级类型、等级、数量和占比的DataFrame。
    """
    # 对需要统计的列进行融化，形成两列：等级类型 和 等级值
    melt_df = df[["参考等级(电网)", "风险评估等级", "风险分析等级"]].melt(
        var_name="等级类型", value_name="等级"
    )
    # 统计数量
    counts = melt_df.value_counts().reset_index(name="数量")
    # 按等级类型计算占比
    counts["占比"] = counts["数量"] / counts.groupby("等级类型")["数量"].transform("sum")

    # 确保等级按照指定顺序排列
    counts["等级"] = pd.Categorical(counts["等级"], categories=RISK_LEVEL_ORDER, ordered=True)
    counts = counts.sort_values(["等级类型", "等级"])

    return counts


def plot_level_distribution(distribution: pd.DataFrame) -> alt.Chart:
    """绘制等级占比对比图，显示三种等级类型的分布差异。"""
    # 创建分组柱状图，便于对比
    chart = (
        alt.Chart(distribution)
        .mark_bar()
        .encode(
            x=alt.X("等级:O", title="等级", sort=RISK_LEVEL_ORDER),
            y=alt.Y("占比:Q", title="占比", axis=alt.Axis(format="%")),
            color=alt.Color("等级类型:N", title="等级类型"),
            xOffset="等级类型:N",
        )
        .properties(width=400, height=300, title="三种等级类型占比对比")
    )
    return chart


def compute_level_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """计算三种等级类型的详细对比表。"""
    level_cols = ["参考等级(电网)", "风险评估等级", "风险分析等级"]
    comparison_data = []

    for col in level_cols:
        dist = df[col].value_counts(normalize=True).sort_index()
        for level in RISK_LEVEL_ORDER:
            if level in dist.index:
                comparison_data.append({"等级类型": col, "等级": level, "占比": dist[level]})
            else:
                comparison_data.append({"等级类型": col, "等级": level, "占比": 0.0})

    comparison_df = pd.DataFrame(comparison_data)
    # 透视表格式，便于对比
    pivot_df = comparison_df.pivot(index="等级", columns="等级类型", values="占比").fillna(0)
    pivot_df = pivot_df.reindex(RISK_LEVEL_ORDER)

    return pivot_df


def compute_mean_by_reference(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """计算各数值列在不同电网参考等级下的平均值。"""
    mean_df = df.groupby("参考等级(电网)")[numeric_cols].mean().reset_index()
    # 确保参考等级按照指定顺序排列
    mean_df["参考等级(电网)"] = pd.Categorical(
        mean_df["参考等级(电网)"], categories=RISK_LEVEL_ORDER, ordered=True
    )
    mean_df = mean_df.sort_values("参考等级(电网)")
    return mean_df


def plot_mean_by_reference(mean_df: pd.DataFrame, numeric_cols: list) -> alt.Chart:
    """绘制数值指标按参考等级的均值柱状图，使用多个Y轴分别展示各指标。"""
    # 将数据转换为长格式以便Altair绘制
    long_df = mean_df.melt(
        id_vars=["参考等级(电网)"], value_vars=numeric_cols, var_name="指标", value_name="平均值"
    )

    # 定义指标顺序，确保综合风险值在最上方
    indicator_order = [
        "综合风险值",
        "动态因子",
        "人员风险值",
        "管理风险值",
        "设备风险值",
        "方法风险值",
        "环境风险值",
    ]

    # 为不同指标定义颜色
    color_mapping = {
        "综合风险值": "#d62728",  # 红色，突出显示
        "动态因子": "#ff7f0e",  # 橙色
        "人员风险值": "#2ca02c",  # 绿色
        "管理风险值": "#1f77b4",  # 蓝色
        "设备风险值": "#9467bd",  # 紫色
        "方法风险值": "#8c564b",  # 棕色
        "环境风险值": "#e377c2",  # 粉色
    }

    # 创建多行子图布局
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "参考等级(电网):O",
                title="参考等级(电网)",
                sort=RISK_LEVEL_ORDER,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("平均值:Q", title="平均值", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "指标:N",
                title="指标",
                sort=indicator_order,
                scale=alt.Scale(
                    domain=indicator_order,
                    range=[color_mapping.get(ind, "#1f77b4") for ind in indicator_order],
                ),
            ),
            row=alt.Row(
                "指标:N",
                title="风险指标",
                sort=indicator_order,
                header=alt.Header(
                    labelAngle=0, labelAlign="left", labelFontSize=12, labelFontWeight="bold"
                ),
            ),
        )
        .properties(
            width=400,
            height=120,
            title=alt.TitleParams(
                text="各风险指标在不同参考等级下的平均值对比", fontSize=16, anchor="start"
            ),
        )
        .resolve_scale(
            y="independent"  # 每个子图使用独立的Y轴刻度
        )
    )
    return chart


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> alt.Chart:
    """绘制带颜色编码的散点图。"""
    chart = (
        alt.Chart(df)
        .mark_circle(opacity=0.6, size=60)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            color=alt.Color(
                f"{color_col}:O",
                title=color_col,
                sort=RISK_LEVEL_ORDER,
                scale=alt.Scale(range=["green", "yellow", "orange", "red"]),
            ),
            tooltip=[x_col, y_col, color_col],
        )
        .interactive()
        .properties(width=500, height=400)
    )
    return chart


def compute_virtual_risk_with_thresholds(
    df: pd.DataFrame, acceptable_threshold: float, low_threshold: float, medium_threshold: float
) -> pd.DataFrame:
    """根据四个等级的阈值划分综合风险值，生成模拟风险等级列。"""

    def classify(value):
        if value < acceptable_threshold:
            return "可接受"
        elif value < low_threshold:
            return "低"
        elif value < medium_threshold:
            return "中"
        else:
            return "高"

    df = df.copy()
    df["模拟风险等级"] = df["综合风险值"].apply(classify)
    return df


def compute_distribution_compare_detailed(
    df: pd.DataFrame, new_col: str = "模拟风险等级"
) -> pd.DataFrame:
    """比较模拟风险等级与电网参考等级的占比差异。"""
    # 参考等级分布
    ref_dist = df["参考等级(电网)"].value_counts(normalize=True)
    new_dist = df[new_col].value_counts(normalize=True)

    # 确保所有等级都包含在内
    comparison_data = []
    for level in RISK_LEVEL_ORDER:
        ref_ratio = ref_dist.get(level, 0)
        sim_ratio = new_dist.get(level, 0)
        comparison_data.append(
            {
                "等级": level,
                "参考等级占比": ref_ratio,
                "模拟等级占比": sim_ratio,
                "差异": sim_ratio - ref_ratio,
            }
        )

    compare_df = pd.DataFrame(comparison_data)
    return compare_df


def plot_distribution_compare_detailed(compare_df: pd.DataFrame) -> alt.Chart:
    """绘制参考等级与模拟等级的占比对比图。"""
    long_df = compare_df.melt(
        id_vars=["等级"], value_vars=["参考等级占比", "模拟等级占比"], var_name="类型", value_name="占比"
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("等级:O", title="等级", sort=RISK_LEVEL_ORDER),
            y=alt.Y("占比:Q", title="占比", axis=alt.Axis(format="%")),
            color=alt.Color("类型:N", title="类别"),
            xOffset="类型:N",
        )
        .properties(width=400, height=300, title="参考等级与模拟等级分布对比")
    )
    return chart


def plot_confusion_matrix(df: pd.DataFrame, new_col: str = "模拟风险等级") -> pd.DataFrame:
    """生成并返回模拟等级与参考等级的混淆矩阵。"""
    matrix = pd.crosstab(df["参考等级(电网)"], df[new_col], normalize="index")
    # 确保行列都按照指定顺序排列
    matrix = matrix.reindex(index=RISK_LEVEL_ORDER, columns=RISK_LEVEL_ORDER, fill_value=0)
    return matrix


def compute_correlation(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """计算相关系数矩阵。"""
    corr = df[numeric_cols].corr()
    return corr


def plot_heatmap(corr: pd.DataFrame) -> alt.Chart:
    """绘制相关性热图。"""
    corr_df = corr.reset_index().melt(id_vars="index", var_name="变量2", value_name="相关系数")
    corr_df = corr_df.rename(columns={"index": "变量1"})
    base = alt.Chart(corr_df).encode(x="变量1:N", y="变量2:N")
    heatmap = base.mark_rect().encode(
        color=alt.Color("相关系数:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="相关系数")
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("相关系数:Q", format=".2f"),
        color=alt.condition(alt.datum.相关系数 > 0.5, alt.value("white"), alt.value("black")),
    )
    return (heatmap + text).properties(width=400, height=400)


def analyze_single_risk_level(df: pd.DataFrame, selected_level: str, numeric_cols: list) -> dict:
    """对单个风险等级进行多维度数据分析。

    Args:
        df: 原始数据框
        selected_level: 选择的参考等级
        numeric_cols: 数值列名列表

    Returns:
        包含分析结果的字典
    """
    # 筛选指定等级的数据
    level_data = df[df["参考等级(电网)"] == selected_level].copy()

    if len(level_data) == 0:
        return {"error": f"没有找到参考等级为'{selected_level}'的数据"}

    analysis_results = {
        "data_count": len(level_data),
        "percentage": len(level_data) / len(df) * 100,
        "numeric_stats": level_data[numeric_cols].describe(),
        "correlation": level_data[numeric_cols].corr(),
    }

    # 如果有作业性质列，分析作业性质分布
    if "作业性质" in level_data.columns:
        analysis_results["work_type_dist"] = level_data["作业性质"].value_counts(normalize=True)

    # 如果有是否跃迁列，分析跃迁情况
    if "是否跃迁" in level_data.columns:
        analysis_results["leap_dist"] = level_data["是否跃迁"].value_counts(normalize=True)
        analysis_results["leap_risk_stats"] = level_data.groupby("是否跃迁")["综合风险值"].describe()

    return analysis_results


def plot_single_level_distribution(
    level_data: pd.DataFrame, numeric_cols: list, selected_level: str
) -> alt.Chart:
    """绘制单个等级下各维度风险值的分布图。"""
    # 转换为长格式
    long_df = level_data[numeric_cols].melt(var_name="风险维度", value_name="风险值")

    # 定义颜色映射
    color_mapping = {
        "综合风险值": "#d62728",
        "动态因子": "#ff7f0e",
        "人员风险值": "#2ca02c",
        "管理风险值": "#1f77b4",
        "设备风险值": "#9467bd",
        "方法风险值": "#8c564b",
        "环境风险值": "#e377c2",
    }

    # 创建箱线图
    chart = (
        alt.Chart(long_df)
        .mark_boxplot(size=50)
        .encode(
            x=alt.X("风险维度:N", title="风险维度", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("风险值:Q", title="风险值"),
            color=alt.Color(
                "风险维度:N",
                scale=alt.Scale(domain=list(color_mapping.keys()), range=list(color_mapping.values())),
                legend=None,
            ),
        )
        .properties(width=600, height=300, title=f"'{selected_level}'等级下各维度风险值分布")
    )

    return chart


def plot_single_level_radar(
    level_data: pd.DataFrame, numeric_cols: list, selected_level: str
) -> alt.Chart:
    """绘制单个等级下风险维度的雷达图（使用极坐标近似）。"""
    # 计算各维度的平均值和标准化
    means = level_data[numeric_cols].mean()

    # 标准化到0-1范围
    normalized_means = (means - means.min()) / (means.max() - means.min())

    # 创建雷达图数据
    radar_data = pd.DataFrame(
        {
            "维度": normalized_means.index,
            "值": normalized_means.values,
            "角度": [i * 360 / len(normalized_means) for i in range(len(normalized_means))],
        }
    )

    # 添加第一个点到末尾以闭合图形
    radar_data = pd.concat([radar_data, radar_data.iloc[[0]]], ignore_index=True)

    # 创建极坐标图
    chart = (
        alt.Chart(radar_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            theta=alt.Theta("角度:Q", scale=alt.Scale(range=[0, 6.28])),
            radius=alt.Radius("值:Q", scale=alt.Scale(range=[0, 100])),
            color=alt.value("#1f77b4"),
            tooltip=["维度:N", "值:Q"],
        )
        .resolve_scale(radius="independent")
        .properties(width=300, height=300, title=f"'{selected_level}'等级风险维度雷达图")
    )

    return chart


def plot_single_level_comparison(df: pd.DataFrame, selected_level: str, numeric_cols: list) -> alt.Chart:
    """绘制选定等级与其他等级的对比图。"""
    # 计算各等级的平均值
    level_means = df.groupby("参考等级(电网)")[numeric_cols].mean().reset_index()

    # 转换为长格式
    long_df = level_means.melt(
        id_vars=["参考等级(电网)"], value_vars=numeric_cols, var_name="风险维度", value_name="平均值"
    )

    # 添加是否为选定等级的标识
    long_df["是否选中"] = long_df["参考等级(电网)"] == selected_level

    # 创建对比图
    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("参考等级(电网):O", title="参考等级", sort=RISK_LEVEL_ORDER),
            y=alt.Y("平均值:Q", title="平均值"),
            color=alt.Color(
                "是否选中:N",
                scale=alt.Scale(range=["lightgray", "#d62728"]),
                legend=alt.Legend(title="等级类型", values=["其他等级", "选中等级"]),
            ),
            facet=alt.Facet("风险维度:N", columns=3, title="风险维度对比"),
            tooltip=["参考等级(电网):N", "风险维度:N", "平均值:Q"],
        )
        .properties(width=120, height=100, title=f"'{selected_level}'等级与其他等级的风险维度对比")
        .resolve_scale(y="independent")
    )

    return chart


def main():
    st.set_page_config(page_title="风险数据分析", layout="wide")
    st.title("风险数据分析仪表盘")
    st.write(
        "利用本应用可以上传Excel文件，进行风险等级占比分析、数值指标与参考等级关系分析、\n"
        "并通过自定义阈值生成模拟风险等级，与电网参考等级进行对比。"
    )

    uploaded_file = st.file_uploader("请选择Excel文件进行分析", type=["xls", "xlsx"])
    if uploaded_file is not None:
        # 提醒用户，文件上传后将被读取为BytesIO
        with st.info("文件已上传，请等待读取……"):
            # 读取数据
            df = load_data(uploaded_file)
            st.success("数据读取成功！")

        # 指定数值列，重点关注综合风险值
        numeric_cols = [
            "综合风险值",
            "人员风险值",
            "设备风险值",
            "方法风险值",
            "环境风险值",
            "管理风险值",
            "动态因子",
        ]

        # ---------- 1. 等级占比分析 ----------
        st.header("一、参考等级与评估等级、分析等级占比对比")
        dist_df = compute_level_distribution(df)
        st.altair_chart(plot_level_distribution(dist_df), use_container_width=True)

        # 显示详细对比表
        st.subheader("等级分布详细对比表")
        comparison_table = compute_level_comparison_table(df)
        st.dataframe(comparison_table.style.format("{:.2%}"))

        # 分析差异
        st.write(
            "**分析说明**：上图和表格展示了三种等级类型的分布差异。\n"
            "- 参考等级(电网)：作为基准标准\n"
            "- 风险评估等级：基于算法评估的结果\n"
            "- 风险分析等级：经过二次模型推理分析的结果\n"
        )

        # ---------- 2. 综合风险值与电网参考等级关系 ----------
        st.header("二、综合风险值与各维度风险值分析")
        st.write(
            "**重点分析**：综合风险值是由人员、设备、方法、环境、管理风险值和动态因子计算得出的核心指标，\n"
            "用于最终的风险等级划分。以下分析展示各指标与参考等级的关系。"
        )

        mean_df = compute_mean_by_reference(df, numeric_cols)
        st.altair_chart(plot_mean_by_reference(mean_df, numeric_cols), use_container_width=True)

        # 重点展示综合风险值
        st.subheader("综合风险值按参考等级分布")
        comprehensive_risk_chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X("参考等级(电网):O", title="参考等级(电网)", sort=RISK_LEVEL_ORDER),
                y=alt.Y("综合风险值:Q", title="综合风险值"),
                color=alt.Color(
                    "参考等级(电网):O",
                    sort=RISK_LEVEL_ORDER,
                    scale=alt.Scale(range=["green", "yellow", "orange", "red"]),
                ),
            )
            .properties(width=500, height=300, title="综合风险值在不同参考等级下的分布")
        )
        st.altair_chart(comprehensive_risk_chart, use_container_width=True)

        st.subheader("散点图分析")
        # 默认显示综合风险值相关的散点图
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("请选择横轴变量", numeric_cols, index=0)  # 默认选择综合风险值
        with col2:
            y_col = st.selectbox("请选择纵轴变量", numeric_cols, index=6)  # 默认选择动态因子
        st.altair_chart(plot_scatter(df, x_col, y_col, "参考等级(电网)"), use_container_width=True)

        # ---------- 3. 模拟风险等级生成与对比 ----------
        st.header("三、模拟风险等级与参考等级对比")
        st.write("通过调整综合风险值的各等级划分阈值，生成模拟风险等级，并与电网参考等级进行对比分析。\n")

        # 获取综合风险值的统计信息
        risk_min = float(df["综合风险值"].min())
        risk_max = float(df["综合风险值"].max())
        risk_q25 = float(df["综合风险值"].quantile(0.25))
        risk_q50 = float(df["综合风险值"].quantile(0.50))
        risk_q75 = float(df["综合风险值"].quantile(0.75))

        st.write(f"综合风险值范围：{risk_min:.2f} - {risk_max:.2f}")

        # 使用三个滑动条分别设置四个等级的阈值
        col1, col2, col3 = st.columns(3)
        with col1:
            acceptable_threshold = st.slider(
                "可接受 → 低风险 阈值", risk_min, risk_max, risk_q25, step=0.01
            )
        with col2:
            low_threshold = st.slider("低风险 → 中风险 阈值", risk_min, risk_max, risk_q50, step=0.01)
        with col3:
            medium_threshold = st.slider("中风险 → 高风险 阈值", risk_min, risk_max, risk_q75, step=0.01)

        # 验证阈值顺序
        if acceptable_threshold >= low_threshold or low_threshold >= medium_threshold:
            st.error("阈值必须按照递增顺序设置：可接受 < 低风险 < 中风险 < 高风险！")
        else:
            df_with_virtual = compute_virtual_risk_with_thresholds(
                df, acceptable_threshold, low_threshold, medium_threshold
            )
            compare_df = compute_distribution_compare_detailed(df_with_virtual)

            st.subheader("分布对比分析")
            st.altair_chart(plot_distribution_compare_detailed(compare_df), use_container_width=True)

            # 显示详细对比数据
            def style_difference(val):
                """根据差异值应用颜色标记"""
                if val > 0.2:  # 相差超过20%
                    return "background-color: #ffcccc"  # 红色
                elif val > 0.1:  # 相差超过10%
                    return "background-color: #fff2cc"  # 黄色
                elif val > 0:  # sim_ratio大于ref_ratio
                    return "background-color: #d4edda"  # 成功色（绿色）
                elif val < -0.2:  # 相差超过-20%
                    return "background-color: #ffcccc"  # 红色
                elif val < -0.1:  # 相差超过-10%
                    return "background-color: #fff2cc"  # 黄色
                else:
                    return ""  # 无特殊颜色

            st.dataframe(
                compare_df.style.format(
                    {"参考等级占比": "{:.2%}", "模拟等级占比": "{:.2%}", "差异": "{:.2%}"}
                ).map(style_difference, subset=["差异"])
            )

            # 计算总体差异
            total_difference = compare_df["差异"].sum()
            st.metric(
                "总体差异度",
                f"{total_difference:.2%}",
                help="所有等级占比差异的总和，越小表示模拟结果越接近参考标准",
            )

            st.subheader("混淆矩阵 (按参考等级归一化)")
            matrix = plot_confusion_matrix(df_with_virtual)
            st.dataframe(matrix.style.format("{:.2%}"))

            # 计算准确率
            accuracy = np.trace(matrix.values) / len(RISK_LEVEL_ORDER)
            st.metric(
                "对角线平均准确率",
                f"{accuracy:.2%}",
                help="混淆矩阵对角线元素的平均值，表示模拟等级与参考等级的一致性",
            )

        # ---------- 4. 补充分析 ----------
        st.header("四、相关性热图")
        corr = compute_correlation(df, numeric_cols)
        st.altair_chart(plot_heatmap(corr), use_container_width=True)

        # 其他分析：作业性质对参考等级和综合风险的影响
        st.subheader("作业性质与风险特征")
        if "作业性质" in df.columns:
            work_type_dist = pd.crosstab(df["作业性质"], df["参考等级(电网)"], normalize="index")
            # 确保列按照指定顺序排列
            work_type_dist = work_type_dist.reindex(columns=RISK_LEVEL_ORDER, fill_value=0)
            st.write("各作业性质对应的参考等级占比：")
            st.dataframe(work_type_dist.style.format("{:.2%}"))

        # 是否跃迁与风险比较
        st.subheader("是否跃迁与综合风险值")
        if "是否跃迁" in df.columns:
            leap_stats = df.groupby("是否跃迁")["综合风险值"].describe().T
            st.dataframe(leap_stats)

        # ---------- 5. 单个风险层级多维度分析 ----------
        st.header("五、单个风险层级多维度分析")
        st.write(
            "选择一个特定的参考等级，深入分析该等级下的数据特征、风险维度分布、\n"
            "相关性模式以及与其他等级的对比情况。"
        )

        # 获取可用的参考等级
        available_levels = df["参考等级(电网)"].unique()
        available_levels = [level for level in RISK_LEVEL_ORDER if level in available_levels]

        # 等级选择器
        selected_level = st.selectbox(
            "请选择要分析的参考等级：",
            available_levels,
            index=0,
            help="选择一个参考等级进行详细的多维度分析",
        )

        if selected_level:
            # 进行单等级分析
            analysis_results = analyze_single_risk_level(df, selected_level, numeric_cols)

            if "error" in analysis_results:
                st.error(analysis_results["error"])
            else:
                # 筛选该等级的数据
                level_data = df[df["参考等级(电网)"] == selected_level]

                # 基本统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "数据量",
                        f"{analysis_results['data_count']:,}",
                        help=f"'{selected_level}'等级的数据条数",
                    )
                with col2:
                    st.metric(
                        "占比",
                        f"{analysis_results['percentage']:.1f}%",
                        help=f"'{selected_level}'等级在总数据中的占比",
                    )
                with col3:
                    avg_risk = level_data["综合风险值"].mean()
                    st.metric(
                        "平均综合风险值",
                        f"{avg_risk:.2f}",
                        help=f"'{selected_level}'等级的平均综合风险值",
                    )

                # 风险维度分布分析
                st.subheader(f"'{selected_level}'等级风险维度分布")

                # 创建两列布局
                col1, col2 = st.columns(2)

                with col1:
                    # 箱线图显示分布
                    distribution_chart = plot_single_level_distribution(
                        level_data, numeric_cols, selected_level
                    )
                    st.altair_chart(distribution_chart, use_container_width=True)

                with col2:
                    # 雷达图显示维度特征
                    radar_chart = plot_single_level_radar(level_data, numeric_cols, selected_level)
                    st.altair_chart(radar_chart, use_container_width=True)

                # 详细统计表
                st.subheader("详细统计信息")
                st.dataframe(
                    analysis_results["numeric_stats"].style.format("{:.3f}"), use_container_width=True
                )

                # 相关性分析
                st.subheader(f"'{selected_level}'等级内部相关性分析")
                level_corr_chart = plot_heatmap(analysis_results["correlation"])
                st.altair_chart(level_corr_chart, use_container_width=True)

                # 与其他等级对比
                st.subheader("与其他等级对比")
                comparison_chart = plot_single_level_comparison(df, selected_level, numeric_cols)
                st.altair_chart(comparison_chart, use_container_width=True)

                # 作业性质分析（如果存在）
                if "work_type_dist" in analysis_results:
                    st.subheader(f"'{selected_level}'等级作业性质分布")
                    work_dist_df = analysis_results["work_type_dist"].reset_index()
                    work_dist_df.columns = ["作业性质", "占比"]

                    work_chart = (
                        alt.Chart(work_dist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("作业性质:N", title="作业性质"),
                            y=alt.Y("占比:Q", title="占比", axis=alt.Axis(format="%")),
                            color=alt.Color("作业性质:N", legend=None),
                            tooltip=["作业性质:N", alt.Tooltip("占比:Q", format=".1%")],
                        )
                        .properties(width=400, height=250, title=f"'{selected_level}'等级作业性质分布")
                    )
                    st.altair_chart(work_chart, use_container_width=True)

                    # 显示详细数据
                    st.dataframe(work_dist_df.style.format({"占比": "{:.2%}"}), use_container_width=True)

                # 跃迁分析（如果存在）
                if "leap_dist" in analysis_results:
                    st.subheader(f"'{selected_level}'等级跃迁情况分析")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**跃迁分布：**")
                        leap_dist_df = analysis_results["leap_dist"].reset_index()
                        leap_dist_df.columns = ["是否跃迁", "占比"]
                        st.dataframe(
                            leap_dist_df.style.format({"占比": "{:.2%}"}), use_container_width=True
                        )

                    with col2:
                        st.write("**跃迁与综合风险值关系：**")
                        leap_risk_stats = analysis_results["leap_risk_stats"]
                        st.dataframe(leap_risk_stats.style.format("{:.3f}"), use_container_width=True)

                # 数据洞察
                st.subheader("数据洞察")
                insights = []

                # 风险值洞察
                max_risk_dim = level_data[numeric_cols[1:]].mean().idxmax()  # 排除综合风险值
                min_risk_dim = level_data[numeric_cols[1:]].mean().idxmin()
                insights.append(f"• **主要风险来源**：{max_risk_dim}的平均值最高")
                insights.append(f"• **相对优势维度**：{min_risk_dim}的平均值最低")

                # 相关性洞察
                corr_matrix = analysis_results["correlation"]
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # 高相关性阈值
                            high_corr_pairs.append(
                                (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                            )

                if high_corr_pairs:
                    insights.append("• **高相关性维度**：")
                    for dim1, dim2, corr_val in high_corr_pairs[:3]:  # 最多显示3个
                        insights.append(f"  - {dim1} 与 {dim2}：相关系数 {corr_val:.2f}")

                # 显示洞察
                for insight in insights:
                    st.write(insight)


if __name__ == "__main__":
    main()
